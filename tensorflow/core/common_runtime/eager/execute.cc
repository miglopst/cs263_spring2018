/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/eager/execute.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

// Initializes the step stats if needed.
void MaybeInitializeStepStats(StepStats* step_stats, EagerContext* ctx) {
  // Lazily initialize the RunMetadata with information about all devices if
  // this is the first call.
  while (step_stats->dev_stats_size() < ctx->devices()->size()) {
    int device_idx = step_stats->dev_stats_size();
    auto* dev_stats = step_stats->add_dev_stats();
    dev_stats->set_device(ctx->devices()->at(device_idx)->name());
  }
}

int StepStatsDeviceIndex(StepStats* step_stats, EagerContext* ctx,
                         Device* device) {
  // Find the current device's index.
  if (device == nullptr) {
    device = ctx->HostCPU();
  }
  for (int i = 0; i < ctx->devices()->size(); ++i) {
    if (ctx->devices()->at(i) == device ||
        ctx->devices()->at(i)->name() == device->name()) {
      return i;
    }
  }
  // TODO(apassos) do not fall back to host CPU if device is unknown.
  return 0;
}

Status ValidateInputTypeAndPlacement(EagerContext* ctx, Device* op_device,
                                     EagerOperation* op, const OpKernel* kernel,
                                     RunMetadata* run_metadata) {
  Device* host_device = ctx->HostCPU();
  const MemoryTypeVector& memtypes = kernel->input_memory_types();
  if (memtypes.size() != op->Inputs().size()) {
    return errors::InvalidArgument("expected ", memtypes.size(),
                                   " inputs, got ", op->Inputs().size());
  }
  for (int i = 0; i < op->Inputs().size(); ++i) {
    const Device* expected_device =
        memtypes[i] == HOST_MEMORY ? host_device : op_device;
    TensorHandle* handle = op->Inputs()[i];
    Device* handle_device = nullptr;
    TF_RETURN_IF_ERROR(handle->Device(&handle_device));
    const Device* actual_device =
        handle_device == nullptr ? host_device : handle_device;
    if (expected_device != actual_device) {
      switch (ctx->GetDevicePlacementPolicy()) {
        case DEVICE_PLACEMENT_SILENT_FOR_INT32:
          // TODO(xpan): See if we could bubble python related error up
          // to python level.
          if (handle->dtype == DT_INT32) {
            // Note: enabling silent copies of int32 tensors to match behavior
            // of graph mode.
            break;
          }
          TF_FALLTHROUGH_INTENDED;
        case DEVICE_PLACEMENT_EXPLICIT:
          return errors::InvalidArgument(
              "Tensors on conflicting devices:"
              " cannot compute ",
              op->Name(), " as input #", i, " was expected to be on ",
              expected_device->name(), " but is actually on ",
              actual_device->name(), " (operation running on ",
              op_device->name(), ")",
              " Tensors can be copied explicitly using .gpu() or .cpu() "
              "methods,"
              " or transparently copied by using tf.enable_eager_execution("
              "device_policy=tfe.DEVICE_PLACEMENT_SILENT). Copying tensors "
              "between devices"
              " may slow down your model");
        case DEVICE_PLACEMENT_WARN:
          LOG(WARNING) << "before computing " << op->Name() << " input #" << i
                       << " was expected to be on " << expected_device->name()
                       << " but is actually on " << actual_device->name()
                       << " (operation running on " << op_device->name()
                       << "). This triggers a copy which can be a performance "
                          "bottleneck.";
          break;
        case DEVICE_PLACEMENT_SILENT:  // Do nothing.
          break;
      }
      // We are only here if the policy is warn or silent copies, so we should
      // trigger a copy.
      auto pre_time = Env::Default()->NowMicros();
      TensorHandle* copied_tensor = nullptr;
      Status status = EagerCopyToDevice(
          handle, ctx, expected_device->name().c_str(), &copied_tensor);
      if (run_metadata != nullptr) {
        auto* step_stats = run_metadata->mutable_step_stats();
        MaybeInitializeStepStats(step_stats, ctx);
        // Record the sending on the source device for now.
        int device_idx = StepStatsDeviceIndex(step_stats, ctx, handle_device);
        auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
        auto* node_stats = dev_stats->add_node_stats();
        node_stats->set_node_name("_Send");
        node_stats->set_all_start_micros(pre_time);
        node_stats->set_op_end_rel_micros(Env::Default()->NowMicros() -
                                          pre_time);
      }
      if (!status.ok()) {
        if (copied_tensor != nullptr) copied_tensor->Unref();
        return errors::Internal("Failed copying input tensor from ",
                                actual_device->name(), " to ",
                                expected_device->name(), " in order to run ",
                                op->Name(), ": ", status.error_message());
      }
      handle->Unref();
      handle = copied_tensor;
      (*op->MutableInputs())[i] = copied_tensor;
    }
    if (handle->dtype != kernel->input_type(i)) {
      return errors::InvalidArgument(
          "cannot compute ", op->Name(), " as input #", i,
          " was expected to be a ", DataTypeString(kernel->input_type(i)),
          " tensor but is a ", DataTypeString(handle->dtype), " tensor");
    }
  }
  return Status::OK();
}

Status SelectDevice(const NodeDef& ndef, EagerContext* ctx, Device** device) {
  DeviceSet ds;
  for (Device* d : *ctx->devices()) {
    ds.AddDevice(d);
  }
  DeviceTypeVector final_devices;
  auto status = SupportedDeviceTypesForNode(ds.PrioritizedDeviceTypeList(),
                                            ndef, &final_devices);
  if (!status.ok()) return status;
  if (final_devices.empty()) {
    return errors::Internal("Could not find valid device for node ",
                            ndef.DebugString());
  }
  for (Device* d : *ctx->devices()) {
    if (d->device_type() == final_devices[0].type_string()) {
      *device = d;
      return Status::OK();
    }
  }
  return errors::Unknown("Could not find a device for node ",
                         ndef.DebugString());
}

#ifdef TENSORFLOW_EAGER_USE_XLA
// Synthesizes and returns a wrapper function over `op`, which must be a
// primitive op (e.g. matmul).
//
// The wrapper function conforms to the function signature expected by
// _XlaLaunchOp, with input params ordered by <constants, (variable) args and
// resources>. For example, if the op has input params <Const1, Arg2, Const3,
// Resource4, Arg5>, they will be reordered to <Const1, Const3, Arg2, Arg5,
// Resource4> as the input params to the synthesized function.
//
// It populates `const_input_types`, `arg_input_types` and
// `op_input_to_func_input` based on the reordering results, that the caller can
// use them to build an _XlaLaunchOp. On error, it returns NULL, and sets
// `status` accordingly.
const FunctionDef* OpToFunction(TFE_Op* op,
                                std::vector<TF_DataType>* const_input_types,
                                std::vector<TF_DataType>* arg_input_types,
                                gtl::FlatMap<int, int>* op_input_to_func_input,
                                TF_Status* status) {
  DCHECK(!op->operation.is_function());

  FunctionDef fdef;

  // Get the OpDef of the op we are trying to encapsulate.
  TFE_Context* ctx = op->operation.ctx;
  const OpRegistrationData* op_data;
  {
    status = ctx->context.FindFunctionOpData(op->operation.Name(), &op_data);
    if (!status.ok()) {
      return nullptr;
    }
  }
  const OpDef& op_def = op_data->op_def;

  OpDef* signature = fdef.mutable_signature();

  // Handle constant inputs.
  const std::unordered_set<string> const_inputs(
      *XlaOpRegistry::CompileTimeConstantInputs(op->operation.Name()));

  // First add place holders for the input args, so that we can refer to them by
  // position in the next loop. Also tally up the resource inputs.
  int num_resource_inputs = 0;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    if (op_def.input_arg(i).type() == DT_RESOURCE) {
      ++num_resource_inputs;
    }
    signature->add_input_arg();
  }

  // Now we map the input params from `op_def` to `signature`, where the param
  // ordering for `signature` is: <constants, args, resources>.
  int const_index = 0;
  int arg_index = const_inputs.size();
  int resource_index = op_def.input_arg_size() - num_resource_inputs;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const OpDef::ArgDef& op_input_arg = op_def.input_arg(i);
    OpDef::ArgDef* func_input_arg = nullptr;
    if (const_inputs.find(op_input_arg.name()) != const_inputs.end()) {
      VLOG(1) << "For const input, mapping op input " << i << " to func input "
              << const_index;
      (*op_input_to_func_input)[i] = const_index;
      func_input_arg = signature->mutable_input_arg(const_index++);
      const_input_types->push_back(
          static_cast<TF_DataType>(op->operation.Inputs()[i]->dtype));
    } else if (op_input_arg.type() == DT_RESOURCE) {
      VLOG(1) << "For resource input, mapping op input " << i
              << " to func input " << resource_index;
      (*op_input_to_func_input)[i] = resource_index;
      func_input_arg = signature->mutable_input_arg(resource_index++);
    } else {
      VLOG(1) << "For arg input, mapping op input " << i << " to func input "
              << arg_index;
      (*op_input_to_func_input)[i] = arg_index;
      func_input_arg = signature->mutable_input_arg(arg_index++);
      arg_input_types->push_back(
          static_cast<TF_DataType>(op->operation.Inputs()[i]->dtype));
    }

    func_input_arg->set_name(op_input_arg.name());
    func_input_arg->set_type(op->operation.Inputs()[i]->dtype);
  }
  VLOG(1) << "Added OpDef Inputs: " << fdef.DebugString();

  // Resources args are at the end of the function input params, and we should
  // have iterated over all of them.
  DCHECK_EQ(signature->input_arg_size(), resource_index);

  // Make the synthesized function's name unique.
  signature->set_name(
      strings::StrCat(op_def.name(), func_id_generator.fetch_add(1)));

  // Add the node def and set its input names to match op_def's names.
  const NodeDef& ndef = op->operation.MutableAttrs()->BuildNodeDef();
  DCHECK_EQ(signature->input_arg_size(), ndef.input_size());
  *fdef.add_node_def() = ndef;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    fdef.mutable_node_def(0)->set_input(i, op_def.input_arg(i).name());
  }
  VLOG(1) << "Added NodeDef: " << fdef.DebugString();

  // Fix the output names and set output types.
  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    OpDef::ArgDef* arg = signature->add_output_arg();
    const OpDef::ArgDef& op_def_arg = op_def.output_arg(i);
    const string& out_tensor_name =
        strings::StrCat(ndef.name(), ":", op_def_arg.name(), ":", 0);
    arg->set_name(op_def_arg.name());
    (*fdef.mutable_ret())[op_def_arg.name()] = out_tensor_name;
    const string& type_attr = op_def_arg.type_attr();
    if (!type_attr.empty()) {
      auto i = ndef.attr().find(type_attr);
      if (i == ndef.attr().end()) {
        status = errors::InvalidArgument(
            strings::StrCat("Could not find attr ", type_attr, " in NodeDef ",
                            ndef.DebugString()));
        return nullptr;
      }
      arg->set_type(i->second.type());
    }
  }
  VLOG(1) << "Fixed Output names and all types: " << fdef.DebugString();

  status = ctx->context.AddFunctionDef(fdef);
  if (!status.ok()) return nullptr;
  const auto ret = ctx->context.FindFunctionDef(signature->name());
  DCHECK(ret != nullptr);
  return ret;
}

// Builds an _XLALaunchOp as a wrapper over 'op', so that 'op' can be executed
// via XLA.
std::unique_ptr<TFE_Op> BuildXlaLaunch(TFE_Op* op, TF_Status* status) {
  VLOG(1) << "Creating _XlaLaunchOp for TFE_Op " << op->operation.Name();
  auto launch_op = std::unique_ptr<TFE_Op>(
      TFE_NewOp(op->operation.ctx, "_XlaLaunch", status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  if (op->operation.device) {
    TFE_OpSetDevice(launch_op.get(), op->operation.device->name().c_str(),
                    status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }

  const FunctionDef* fdef;
  { fdef = op->operation.ctx->FindFunctionDef(op->operation.Name()); }
  std::vector<TF_DataType> const_input_types;
  std::vector<TF_DataType> arg_input_types;
  gtl::FlatMap<int, int> op_input_to_func_input;
  if (fdef == nullptr) {
    // See if this is a primitive op, and if so create a function for it, so
    // that _XlaLaunchOp can access it.
    fdef = OpToFunction(op, &const_input_types, &arg_input_types,
                        &op_input_to_func_input, status);
    if (!status.ok()) return nullptr;
  } else {
    // TODO(hongm): XlaOpRegistry::CompileTimeConstantInputs() does not work for
    // functions, so we need to find another way to handle constant inputs.
    for (int i = const_input_types.size();
         i < fdef->signature().input_arg_size(); ++i) {
      VLOG(1) << "Adding Targs from input arg " << i;
      const OpDef::ArgDef& arg = fdef->signature().input_arg(i);
      arg_input_types.push_back(static_cast<TF_DataType>(arg.type()));
    }
  }
  DCHECK(fdef != nullptr);

  // Copy inputs and their devices.
  // Since input param reordering may have occurred between `op` and `launch_op`
  // via `op_input_to_func_input`, adjust the actual inputs accordingly.
  *launch_op->operation.MutableInputs() = op->operation.Inputs();
  for (TensorHandle* h : launch_op->operation.Inputs()) {
    h->Ref();
  }
  if (!op_input_to_func_input.empty()) {
    DCHECK_EQ(op->operation.Inputs().size(), op_input_to_func_input.size());
    for (int i = 0; i < op_input_to_func_input.size(); ++i) {
      VLOG(1) << "mapping op input " << i << " to func input "
              << op_input_to_func_input[i];

      (*launch_op->operation.MuableInputs())[op_input_to_func_input[i]] =
          op->operation.Inputs()[i];
    }
  }
  launch_op->operation.MutableAttrs()->NumInputs(op->operation.Inputs().size());

  TFE_OpSetAttrTypeList(launch_op.get(), "Tconstants", const_input_types.data(),
                        const_input_types.size());

  // Set Targs and Nresources attrs.
  TFE_OpSetAttrTypeList(launch_op.get(), "Targs", arg_input_types.data(),
                        arg_input_types.size());
  const int num_resource_inputs = fdef->signature().input_arg_size() -
                                  const_input_types.size() -
                                  arg_input_types.size();
  TFE_OpSetAttrInt(launch_op.get(), "Nresources", num_resource_inputs);

  // Set Tresults attr.
  std::vector<TF_DataType> tresults;
  for (const OpDef::ArgDef& arg : fdef->signature().output_arg()) {
    tresults.push_back(static_cast<TF_DataType>(arg.type()));
  }
  TFE_OpSetAttrTypeList(launch_op.get(), "Tresults", tresults.data(),
                        tresults.size());

  // Set function attr.
  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(fdef->signature().name());
  launch_op->attrs.Set("function", attr_value);

  return launch_op;
}
#endif  // TENSORFLOW_EAGER_USE_XLA

}  // namespace

Status EagerExecute(EagerOperation* op,
                    gtl::InlinedVector<TensorHandle*, 2>* retvals,
                    int* num_retvals) {
  EagerContext* ctx = op->EagerContext();
  auto status = ctx->GetStatus();
  if (!status.ok()) return status;
#ifdef TENSORFLOW_EAGER_USE_XLA
  std::unique_ptr<TFE_Op> xla_launch_op;
  if (op->UseXla() && op->Name() != "_XlaLaunch") {
    xla_launch_op = BuildXlaLaunch(op, status);
    if (!status.ok()) return status;
    op = xla_launch_op.get();
  }
#endif  // TENSORFLOW_EAGER_USE_XLA
  // Ensure all resource-touching ops run in the device the resource is,
  // regardless of anything else that has been specified. This is identical to
  // the graph mode behavior.
  for (int i = 0; i < op->Inputs().size(); ++i) {
    Device* input_op_device = nullptr;
    status = op->Inputs()[i]->OpDevice(&input_op_device);
    if (!status.ok()) return status;
    VLOG(2) << "for op " << op->Name() << " input " << i << " "
            << DataTypeString(op->Inputs()[i]->dtype) << " "
            << (input_op_device == nullptr ? "cpu" : input_op_device->name())
            << " " << (op->Device() == nullptr ? "cpu" : op->Device()->name());
    if (op->Inputs()[i]->dtype == DT_RESOURCE &&
        (input_op_device != op->Device() || input_op_device == nullptr)) {
      Device* d = input_op_device == nullptr ? ctx->HostCPU() : input_op_device;
      VLOG(1) << "Changing device of operation " << op->Name() << " to "
              << d->name() << " because input #" << i
              << " is a resource in this device.";
      op->SetDevice(d);
    }
  }
  Device* device = op->Device();

  Fprint128 cache_key = op->MutableAttrs()->CacheKey(
      device == nullptr ? "unspecified" : device->name());
  KernelAndDevice* kernel = ctx->GetCachedKernel(cache_key);
  if (kernel == nullptr) {
    const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
    if (device == nullptr) {
      status = SelectDevice(ndef, ctx, &device);
      if (!status.ok()) return status;
    }
    CHECK(device != nullptr);
    if (ctx->LogDevicePlacement()) {
      LOG(INFO) << "Executing op " << ndef.op() << " in device "
                << device->name();
    }
    kernel = new KernelAndDevice(ctx->GetRendezvous());
    // Knowledge of the implementation of Init (and in-turn
    // FunctionLibraryRuntime::CreateKernel) tells us that ctx->func_lib_def
    // will be accessed, so grab on to the lock.
    // See WARNING comment in Execute (before kernel->Run) - would be nice to
    // rework to avoid this subtlety.
    tf_shared_lock l(*ctx->FunctionsMu());
    status = KernelAndDevice::Init(ndef, ctx->func_lib(device), kernel);
    if (!status.ok()) {
      delete kernel;
      return status;
    }
    // Update output_dtypes inside `kernel`.
    const OpDef* op_def = nullptr;
    const FunctionDef* function_def = ctx->FuncLibDef()->Find(ndef.op());
    if (function_def != nullptr) {
      op_def = &(function_def->signature());
    }
    if (op_def == nullptr) {
      status = OpDefForOp(ndef.op().c_str(), &op_def);
      if (!status.ok()) return status;
    }
    DataTypeVector input_dtypes;
    status = InOutTypesForNode(ndef, *op_def, &input_dtypes,
                               kernel->mutable_output_dtypes());
    if (!status.ok()) return status;
    ctx->AddKernelToCache(cache_key, kernel);
  }
  const DataTypeVector& output_dtypes = kernel->output_dtypes();
  const int output_dtypes_size = static_cast<int>(output_dtypes.size());
  if (output_dtypes_size > *num_retvals) {
    return errors::InvalidArgument("Expecting ", output_dtypes.size(),
                                   " outputs, but *num_retvals is ",
                                   *num_retvals);
  }
  *num_retvals = output_dtypes_size;
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }
  status = ValidateInputTypeAndPlacement(
      ctx, device, op, kernel->kernel(),
      ctx->ShouldStoreMetadata() ? ctx->RunMetadataProto() : nullptr);
  if (!status.ok()) return status;
  std::unique_ptr<NodeExecStats> maybe_stats;
  if (ctx->ShouldStoreMetadata()) {
    maybe_stats.reset(new NodeExecStats);
    maybe_stats->set_node_name(op->Name());
    maybe_stats->set_all_start_micros(Env::Default()->NowMicros());
    maybe_stats->set_op_start_rel_micros(0);
    maybe_stats->set_scheduled_micros(Env::Default()->NowMicros());
    // TODO(apassos) track referenced tensors
  }
  retvals->resize(*num_retvals);
  if (ctx->Async()) {
    // Note that for async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(agarwal): Consider executing "cheap" kernels inline for performance.
    tensorflow::uint64 id = ctx->NextId();
    for (int i = 0; i < *num_retvals; ++i) {
      (*retvals)[i] = new TensorHandle(id, output_dtypes[i], ctx);
    }
    EagerNode* node =
        new ExecuteNode(id, ctx, op->Device(), op->Inputs(), kernel,
                        maybe_stats.release(), output_dtypes, *retvals);
    ctx->ExecutorAdd(node);
  } else {
    // Execute checks if retvals[i] is nullptr or not to figure if it needs to
    // allocate it.
    status = EagerExecute(ctx, op->Device(), op->Inputs(), kernel,
                          maybe_stats.get(), retvals->data(), *num_retvals);
  }

  return status;
}

Status EagerExecute(EagerContext* ctx, Device* device,
                    const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
                    KernelAndDevice* kernel, NodeExecStats* maybe_stats,
                    TensorHandle** retvals, int num_retvals) {
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }

  std::vector<Tensor> outputs(1);
  const MemoryTypeVector* output_memory_types = nullptr;
  output_memory_types = &kernel->kernel()->output_memory_types();
  std::vector<Tensor> inputs(op_inputs.size());
  for (int i = 0; i < op_inputs.size(); ++i) {
    const Tensor* input_tensor = nullptr;
    TF_RETURN_IF_ERROR(op_inputs[i]->Tensor(&input_tensor));
    inputs[i] = *input_tensor;
  }
  // WARNING: kernel->Run utilizes the FunctionLibraryRuntime
  // (ctx->func_lib(device)), which in turn holds a pointer to func_lib_def.
  // But knowledge of the implementation
  // of FunctionLibraryRuntime tells us that func_lib_def is not accessed by
  // FunctionLibraryRuntime::Run(), so there is no thread-safety concern here.
  // This is quite subtle. Re-work things to make this better?  (Would it make
  // sense for FunctionLibraryRuntime to ensure thread-safe access to
  // FunctionLibraryDefinition?).  TODO(apassos) figure out how to record stats
  // for ops which are a part of functions.
  // TODO(agarwal): change Run to take vector of handles ?
  TF_RETURN_IF_ERROR(kernel->Run(&inputs, &outputs, maybe_stats));
  if (maybe_stats != nullptr) {
    maybe_stats->set_op_end_rel_micros(Env::Default()->NowMicros() -
                                       maybe_stats->all_start_micros());
    mutex_lock ml(*ctx->MetadataMu());
    if (ctx->ShouldStoreMetadata()) {
      auto* step_stats = ctx->RunMetadataProto()->mutable_step_stats();
      // Lazily initialize the RunMetadata with information about all devices if
      // this is the first call.
      while (step_stats->dev_stats_size() < ctx->devices()->size()) {
        step_stats->add_dev_stats();
      }
      // Find the current device's index.
      int device_idx = 0;
      for (int i = 0; i < ctx->devices()->size(); ++i) {
        if (ctx->devices()->at(i) == device) {
          device_idx = i;
          break;
        }
      }
      // Populate the device stats for this device.
      auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
      dev_stats->set_device(device->name());
      *dev_stats->add_node_stats() = *maybe_stats;
    }
  }
  DCHECK_EQ(num_retvals, outputs.size());
  Device* op_device = device;
  for (int i = 0; i < num_retvals; ++i) {
    Device* d = op_device;
    if (d != nullptr && output_memory_types != nullptr &&
        (*output_memory_types)[i] == HOST_MEMORY) {
      d = nullptr;
    }
    if (retvals[i] == nullptr) {
      retvals[i] = new TensorHandle(outputs[i], d, op_device, ctx);
    } else {
      retvals[i]->SetTensorAndDevice(outputs[i], d, op_device);
    }
  }
  return Status::OK();
}

Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         const char* device_name, TensorHandle** result) {
  TF_RETURN_IF_ERROR(ctx->GetStatus());
  Device* dstd = ctx->HostCPU();
  if (device_name != nullptr && strlen(device_name) > 0) {
    TF_RETURN_IF_ERROR(ctx->device_mgr()->LookupDevice(device_name, &dstd));
  }
  if (ctx->Async()) {
    // Note that `h` may not be currently ready. However execution order will
    // make sure that `h` is ready before the copy is actually done.
    CopyToDeviceNode* node = new CopyToDeviceNode(h, dstd, ctx);
    TensorHandle* output = node->dst();
    // Note that calling Add makes `node` accessible by the EagerExecutor
    // thread. So further accesses need to be thread-safe.
    ctx->ExecutorAdd(node);
    *result = output;
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, dstd, result));
    return Status::OK();
  }
}

}  // namespace tensorflow
