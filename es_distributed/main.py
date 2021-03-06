# Modified from main.py from OpenAI's evolutionary-strategies-starter project
import errno
import json
import logging
import multiprocessing
import os
import sys

import click

from .dist import RelayClient
from .es import run_master, run_worker, SharedNoiseTable


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=$(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)


@cli.command()
@click.option('--exp_file')
@click.option('--master_socket_path', required=True)
@click.option('--log_dir')
def master(exp_file, master_socket_path, log_dir):
    # Start the master
    with open(exp_file, 'r') as f:
        exp = json.loads(f.read())
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)
    run_master({'unix_socket_path': master_socket_path}, log_dir, exp)


@cli.command()
@click.option('--master_socket_path', required=True)
@click.option('--relay_socket_path', required=True)
@click.option('--num_workers', type=int, default=0)
def workers(master_socket_path, relay_socket_path, num_workers):
    # Start the relay
    master_redis_cfg = {'unix_socket_path': master_socket_path}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return
    # Start the workers
    noise = SharedNoiseTable()   # Workers share the same noise
    num_workers = num_workers if num_workers else multiprocessing.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            run_worker(relay_redis_cfg, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()