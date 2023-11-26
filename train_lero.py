import argparse

import os
import json
import socket
from collections import defaultdict
from multiprocessing import Pool

from helper import run
from utils import do_run_query, explain_query
from config import SEP, LOG_PATH, LERO_SERVER_PORT, LERO_SERVER_HOST, LERO_SERVER_PATH, LERO_DUMP_CARD_FILE, PG_DB_PATH


class CardinalityGuidedEntity:
    def __init__(self, score, card_str) -> None:
        self.score = score
        self.card_str = card_str

    def get_score(self):
        return self.score


def create_training_file(training_data_file, *latency_files):
    pair_dict = defaultdict(lambda: [])

    for latency_file in latency_files:
        with open(latency_file, 'r') as file:
            for _line in file.readlines():
                key, value = _line.strip().split(SEP)
                pair_dict[key].append(value)

    training_data = [SEP.join(values) for values in pair_dict.values() if len(values) > 1]

    with open(training_data_file, 'w') as f2:
        f2.write("\n".join(training_data))


class PgHelper:
    def __init__(self, _queries, _output_query_latency_file) -> None:
        self.queries = _queries
        self.output_query_latency_file = _output_query_latency_file

    def start(self, _pool_num):
        pool = Pool(_pool_num)
        for fp, q in self.queries:
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, "pg"))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


class LeroHelper:
    def __init__(self, _queries, _query_num_per_chunk, _output_query_latency_file,
                 _test_queries, _model_prefix, _top_k) -> None:
        self.queries = _queries
        self.query_num_per_chunk = _query_num_per_chunk
        self.output_query_latency_file = _output_query_latency_file
        self.test_queries = _test_queries
        self.model_prefix = _model_prefix
        self.top_k = _top_k
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def start(self, _pool_num):
        lero_chunks = list(self.chunks(self.queries, self.query_num_per_chunk))

        run_args = self.get_run_args()
        for c_idx, chunk in enumerate(lero_chunks):
            pool = Pool(_pool_num)
            for fp, q in chunk:
                self.run_pairwise(q, fp, run_args, self.output_query_latency_file,
                                  self.output_query_latency_file + "_exploratory", pool)
            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()

            model_name = self.model_prefix + "_" + str(c_idx)
            self.retrain(model_name)
            self.test_benchmark(self.output_query_latency_file + "_" + model_name)

    def retrain(self, model_name):
        training_data_file = self.output_query_latency_file + ".training"
        create_training_file(training_data_file, self.output_query_latency_file,
                             self.output_query_latency_file + "_exploratory")
        print("retrain Lero model:", model_name, "with file", training_data_file)

        run(os.path.abspath(training_data_file), model_name)

        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type": "load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        print(reply_json)
        os.system("sync")

    def test_benchmark(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
            do_run_query(q, fp, run_args, output_file, True, "lero")

    def get_run_args(self):
        run_args = ["SET enable_lero TO True"]
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = ["SET lero_joinest_fname TO '" + card_file_name + "'"]
        return run_args

    def run_pairwise(self, q, fp, run_args, _output_query_latency_file, exploratory_query_latency_file, pool):
        explain_query(q, run_args, "EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) ")  # check this!!!
        policy_entities = []
        with open(self.lero_card_file_path, 'r') as file:
            lines = file.readlines()
            lines = [_line.strip().split(";") for _line in lines]
            for _line in lines:
                policy_entities.append(CardinalityGuidedEntity(float(_line[1]), _line[0]))

        policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
        policy_entities = policy_entities[:self.top_k]

        i = 0
        for entity in policy_entities:
            if isinstance(entity, CardinalityGuidedEntity):
                card_str = "\n".join(entity.card_str.strip().split(" "))
                # ensure that the cardinality file will not be changed during planning
                card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                with open(card_file_path, "w") as card_file:
                    card_file.write(card_str)

                output_file = _output_query_latency_file if i == 0 else exploratory_query_latency_file
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name),
                                                     output_file, True, "lero"))
                i += 1

    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type": "predict", "Plan": plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--test_query_path",
                        metavar="PATH",
                        help="Load the test queries")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--query_num_per_chunk", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int)
    parser.add_argument("--topK", type=int)
    args = parser.parse_args()

    query_path = args.query_path
    print("Load queries from ", query_path)
    queries = []
    with open(query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(SEP)
            queries.append((arr[0], arr[1]))
    print("Read", len(queries), "training queries.")

    output_query_latency_file = args.output_query_latency_file
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 10
    if args.pool_num:
        pool_num = args.pool_num
    print("pool_num:", pool_num)

    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if args.algo:
        assert args.algo.lower() in ALGO_LIST
        algo = args.algo.lower()
    print("algo:", algo)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        test_queries = []
        if args.test_query_path is not None:
            with open(args.test_query_path, 'r') as f:
                for line in f.readlines():
                    arr = line.strip().split(SEP)
                    test_queries.append((arr[0], arr[1]))
        print("Read", len(test_queries), "test queries.")

        query_num_per_chunk = args.query_num_per_chunk
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if args.model_prefix:
            model_prefix = args.model_prefix
        print("model_prefix:", model_prefix)

        topK = 5
        if args.topK is not None:
            topK = args.topK
        print("topK", topK)

        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK)
        helper.start(pool_num)

# python train_model.py --query_path stats_test.txt  --algo pg --output_query_latency_file pg_stats_test.log

# python train_model.py --query_path stats_train.txt --test_query_path stats_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_stats.log --model_prefix stats_model --topK 3
