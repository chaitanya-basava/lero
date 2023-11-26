import json
import numpy as np

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort",
            "Limit"] + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


def json_str_to_json_obj(json_data) -> dict:
    json_obj = json.loads(json_data)
    return json_obj[0] if isinstance(json_obj, list) and len(json_obj) == 1 else json_obj


class Preprocessor:
    def __init__(self) -> None:
        self.normalizer = None
        self.feature_parser = None
        self.input_feature_dim = None

    def fit(self, trees):
        def compute_min_max(x):
            x = np.log(np.array(x) + 1)
            return np.min(x), np.max(x)

        exec_times, startup_costs, total_costs, rows, input_relations, rel_types = [], [], [], [], set(), set()

        def parse_node(node):
            startup_costs.append(node.get("Startup Cost"))
            total_costs.append(node.get("Total Cost"))
            rows.append(node.get("Plan Rows"))
            rel_types.add(node.get("Node Type"))
            if "Relation Name" in node:
                input_relations.add(node.get("Relation Name"))

            for child in node.get("Plans", []):
                parse_node(child)

        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj.get("Execution Time")))
            parse_node(json_obj.get("Plan"))

        mins, maxs = {}, {}
        for name, values in [("Startup Cost", startup_costs), ("Total Cost", total_costs),
                             ("Plan Rows", rows), ("Execution Time", exec_times)]:
            if name == "Execution Time" and len(values) == 0:
                continue
            mins[name], maxs[name] = compute_min_max(values)

        self.normalizer = Normalizer(mins, maxs)
        self.feature_parser = FeatureParser(self.normalizer, list(input_relations))

    def transform(self, trees):
        local_features, labels = [], []

        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            plan = json_obj["Plan"]
            if not isinstance(plan, dict):
                plan = json.loads(plan)
                json_obj["Plan"] = plan

            local_features.append(self.feature_parser.extract_feature(plan))

            exec_time = json_obj.get("Execution Time")
            if exec_time is not None:
                exec_time = float(exec_time)
                if self.normalizer.contains("Execution Time"):
                    exec_time = self.normalizer.norm(exec_time, "Execution Time")
                labels.append(exec_time)
            else:
                labels.append(None)

        return local_features, labels


class PlanNode:
    def __init__(self, node_type: np.ndarray,
                 rows: float, width: int,
                 left_child, right_child,
                 input_tables: list, encoded_input_tables: list):
        self.node_type = node_type
        self.rows = rows
        self.width = width
        self.left_child: PlanNode = left_child
        self.right_child: PlanNode = right_child
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables

    def get_feature(self):
        return np.hstack((self.node_type, np.array(self.encoded_input_tables), np.array([self.width, self.rows])))

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def get_subtrees(self):
        subtrees = [self]
        if self.left_child is not None:
            subtrees.extend(self.left_child.get_subtrees())
        if self.right_child is not None:
            subtrees.extend(self.right_child.get_subtrees())
        return subtrees


class Normalizer:
    def __init__(self, mins: dict, maxs: dict):
        self.min_vals = mins
        self.max_vals = maxs

    def _validate_feature_name(self, feature_name):
        if not self.contains(feature_name):
            raise ValueError(f"Feature '{feature_name}' is not present in the normalizer.")

    def norm(self, x, feature_name):
        self._validate_feature_name(feature_name)
        min_val = self.min_vals[feature_name]
        max_val = self.min_vals[feature_name]
        return (np.log(x + 1) - min_val) / (max_val - min_val)

    def contains(self, feature_name):
        return feature_name in self.min_vals and feature_name in self.max_vals


class FeatureParser:
    def __init__(self, normalizer: Normalizer, input_relations: list):
        self.normalizer = normalizer
        self.input_relations = input_relations

    def extract_feature(self, plan_json) -> PlanNode:
        child_relations = []
        left_child, right_child = self._extract_children_features(plan_json, child_relations)

        node_type = op_to_one_hot(plan_json['Node Type'])
        rows = self.normalizer.norm(float(plan_json['Plan Rows']), 'Plan Rows')
        width = int(plan_json['Plan Width'])

        if plan_json['Node Type'] in SCAN_TYPES:
            child_relations.extend([plan_json["Relation Name"]])

        return PlanNode(node_type, rows, width, left_child, right_child,
                        child_relations, self._encode_relation_names(child_relations))

    def _extract_children_features(self, plan_json, child_relations):
        left_child = right_child = None

        if 'Plans' in plan_json:
            assert 0 < len(plan_json['Plans']) <= 2
            left_child = self.extract_feature(plan_json['Plans'][0])
            child_relations.extend(left_child.input_tables)

            if len(plan_json['Plans']) == 2:
                right_child = self.extract_feature(plan_json['Plans'][1])
            else:
                right_child = PlanNode(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0,
                                       None, None, [], self._encode_relation_names([]))

            child_relations.extend(right_child.input_tables)

        return left_child, right_child

    def _encode_relation_names(self, relation_names: list) -> list:
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in relation_names:
            idx = self.input_relations.index(name) if name in self.input_relations else -1
            encode_arr[idx] += 1

        return encode_arr.tolist()


def op_to_one_hot(op_name):
    one_hot = np.zeros(len(OP_TYPES))
    index = OP_TYPES.index(op_name) if op_name in OP_TYPES else OP_TYPES.index(UNKNOWN_OP_TYPE)
    one_hot[index] = 1
    return one_hot
