from es_distributed.main import master, workers
import os


if __name__ == "__main__":
    dir_path = os.getcwd()

    master(exp_file=dir_path+"configurations/pong.json",
           master_socket_path="/tmp/es_redis_master.sock",
           log_dir=dir_path+"results/pong2/")

    workers(master_socket_path="/tmp/es_redis_master.sock",
           relay_socket_path="/tmp/es_redis_relay.sock",
           num_workers=8)