from es_distributed.main import master

if __name__ == "__main__":
    master(exp_str=None, exp_file="/Users/rshen/github/dl-project/es_distributed/go.json",
           master_socket_path="/var/run/redis/redis.sock", log_dir="~/test/")