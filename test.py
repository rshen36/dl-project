from es_distributed.main import workers

if __name__ == "__main__":
    workers(master_socket_path="/tmp/es_redis_master.sock",
            relay_socket_path="/tmp/es_redis_relay.sock",
            num_workers=1)