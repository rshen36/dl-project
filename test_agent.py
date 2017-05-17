from es_distributed.main import master, workers

if __name__ == "__main__":
    master(exp_file="configurations/pong.json",
           master_socket_path="/tmp/es_redis_master.sock",
           log_dir="results/pong2/")

    workers(master_socket_path="/tmp/es_redis_master.sock",
           relay_socket_path="/tmp/es_redis_relay.sock",
           num_workers=8)