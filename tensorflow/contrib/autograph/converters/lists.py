# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converter for list operations.

This includes converting Python lists to TensorArray/TensorList.
"""

# TODO(mdan): Elaborate the logic here.
# TODO(mdan): Does it even make sense to attempt to try to use TAs?
# The current rule (always convert to TensorArray) is naive and insufficient.
# In general, a better mechanism could look like:
#   * convert to TensorList by default
#   * leave as Python list if the user explicitly forbids it
#   * convert to TensorArray only when complete write once behavior can be
#     guaranteed (e.g. list comprehensions)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.framework import dtypes


class ListTransformer(transformer.Base):
  """Converts lists and related operations to their TF counterpart."""

  def _empty_list(self, node):
    if not anno.hasanno(node, 'element_type'):
      raise NotImplementedError(
          'type inference for empty lists is not yet supported; '
          'use set_element_type(<list>, <dtype>) to continue')
    dtype = anno.getanno(node, 'element_type')
    if not isinstance(dtype, dtypes.DType):
      # TODO(mdan): Allow non-TF dtypes?
      # That would be consistent with the dynamic dispatch pattern, but
      # we must make sure that doesn't become confusing.
      raise NotImplementedError('element type "%s" not yet supported' % dtype)

    dtype_name = dtype.name
    # TODO(mdan): Does it ever make sense not to use tensor lists?
    template = """
      tf.TensorArray(tf.dtype_name, size=0, dynamic_size=True)
    """
    return templates.replace_as_expression(template, dtype_name=dtype_name)

  def _pre_populated_list(self, node):
    raise NotImplementedError('pre-populated lists')

  def visit_Expr(self, node):
    node = self.generic_visit(node)
    if isinstance(node.value, gast.Call):
      call_node = node.value

      if not anno.hasanno(call_node.func, anno.Basic.QN):
        return node
      qn = anno.getanno(call_node.func, anno.Basic.QN)

      if qn.qn[-1] == 'append' and (len(call_node.args) == 1):
        template = """
          target = ag__.utils.dynamic_list_append(target, element)
        """
        node = templates.replace(
            template,
            target=qn.parent.ast(),
            element=call_node.args[0])
    return node

  def _replace_list_constructors(self, targets, values):
    for target in targets:
      if (isinstance(target, (gast.Tuple, gast.List)) and
          isinstance(values, (gast.Tuple, gast.List))):
        n_targets = len(target.elts)
        for i in range(n_targets):
          target_el, value_el = target.elts[i], values.elts[i]
          values.elts[i] = self._replace_list_constructors(
              (target_el,), value_el)
        return values
      if isinstance(values, gast.List):
        if values.elts:
          return self._pre_populated_list(values)
        else:
          return self._empty_list(values)
    return values

  def visit_Assign(self, node):
    node = self.generic_visit(node)

    # Only convert lists when they are assigned to a variable, e.g.:
    #   l = []
    # TODO(mdan): A similar pattern exists in type_info.py
    # We should add a generic "unpack_assignment" function to the base
    # transformer, that has the same effect as applying some logic to the SSA
    # form.
    node.value = self._replace_list_constructors(node.targets, node.value)
    return node


def transform(node, context):
  return ListTransformer(context).visit(node)
