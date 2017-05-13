from refactoring.master import start_master

if __name__ == "__main__":
    start_master(exp_file="/Users/rshen/github/dl-project/refactoring/go.json",
                 master_socket_path="/tmp/es_redis_master.sock",
                 log_dir="/Users/rshen/github/dl-project/test/")