# Modified from viz.py from OpenAI's evolutionary-strategies-starter project
import click


@click.command()
@click.argument('env_id')
@click.argument('policy_file')
@click.argument('policy_type')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policy_file, policy_type, record, stochastic, extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    from es_distributed import policies
    import numpy as np

    env = gym.make(env_id)
    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    with tf.Session():
        pi = getattr(policies, policy_type).Load(policy_file, extra_kwargs=extra_kwargs)
        while True:
            rews, t = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
            print('return={:.4f} len={}'.format(rews.sum(), t))

            if record:
                env.close()
                return


if __name__ == '__main__':
    main()