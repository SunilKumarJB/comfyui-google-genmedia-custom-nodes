# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List, Tuple, Union

from .custom_exceptions import APIInputError
from .logger import get_node_logger

logger = get_node_logger(__name__)

class JSONParse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("JSON",)
    FUNCTION = "parse"
    CATEGORY = "Google AI/JSON"

    def parse(self, json_string: str) -> Tuple[Any]:
        try:
            parsed = json.loads(json_string)
            return (parsed,)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise APIInputError(f"Invalid JSON string: {e}")

class JSONGetValue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON",)
    FUNCTION = "get_value"
    CATEGORY = "Google AI/JSON"

    def get_value(self, json_data: Union[Dict, List], key: str) -> Tuple[Any]:
        if isinstance(json_data, dict):
            if key in json_data:
                return (json_data[key],)
            else:
                logger.warning(f"Key '{key}' not found in JSON object.")
                return (None,)
        elif isinstance(json_data, list):
            try:
                idx = int(key)
                if 0 <= idx < len(json_data):
                    return (json_data[idx],)
                else:
                    logger.warning(f"Index '{idx}' out of bounds for JSON array.")
                    return (None,)
            except ValueError:
                logger.warning(f"Cannot use key '{key}' on JSON array (requires integer index).")
                return (None,)
        else:
            logger.warning(f"Cannot get value from primitive type: {type(json_data)}")
            return (None,)

class JSONToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("JSON",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "Google AI/JSON"

    def to_string(self, json_data: Any) -> Tuple[str]:
        if isinstance(json_data, (dict, list)):
            return (json.dumps(json_data, indent=2),)
        else:
            return (str(json_data),)

class JSONIterate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_array": ("JSON",),
            }
        }

    RETURN_TYPES = ("JSON",)
    FUNCTION = "iterate"
    CATEGORY = "Google AI/JSON"

    def iterate(self, json_array: Any) -> Tuple[List[Any]]:
        if isinstance(json_array, list):
            return (json_array,)
        else:
            logger.warning(f"Input is not a list, cannot iterate: {type(json_array)}")
            return ([json_array],)

NODE_CLASS_MAPPINGS = {
    "JSONParse": JSONParse,
    "JSONGetValue": JSONGetValue,
    "JSONToString": JSONToString,
    "JSONIterate": JSONIterate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONParse": "JSON Parse",
    "JSONGetValue": "JSON Get Value",
    "JSONToString": "JSON To String",
    "JSONIterate": "JSON Iterate",
}

