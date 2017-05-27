from es_distributed.main import master, workers

if __name__ == "__main__":
    master(master_socket_path="/tmp/es_redis_master.sock",
           exp_file="/Users/rshen/github/dl-project/configurations/pong.json",
           log_dir="/Users/rshen/github/dl-project/logs/test/")

    workers(master_socket_path="/tmp/es_redis_master.sock",
            relay_socket_path="/tmp/es_redis_relay.sock",
            num_workers=1)