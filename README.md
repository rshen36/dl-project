Repository for Richard Shen's project for Advanced Topics in Deep Learning S17

Supervisor: Lorenzo Torresani   
Contributors: Richard Shen  
Project: Evolution Strategies as an At-Scale Alternative to Reinforcement Learning

To run my implementation of the ES algorithm or A3C algorithm, use the code on the master branch.
To run my implementation of the combined algorithm, use the code on the es-a3c branch.

Regardless of implementation to run, use the following commands in the project directory to run my code:

redis-server redis_config/redis_master.conf
redis-server redis_config/redis_local_mirror.conf
python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --exp_file configurations/pong.json --log_dir example/
python -m es_distributed.main workers --master_socket_path /tmp/es_redis.master.sock --relay_socket_path /tmp/es_redis_master.sock

For the above commands, change the argument for --exp_file when starting the master process to change the environment on which to run the code. Note that, for code on the es-a3c branch, only the configurations files for Pong and Breakout have been maintained, and the configuration file for Go on that branch will cause an error. For code on the master branch, all configuration files should be perfectly fine to pass as an argument.
As well, change the argument for --log_dir to change the location of the directory to which to output the logs of the results, the events file (for visualizing training on Tensorboard), and the snapshots of the policy.
You may also pass an additional argument --num_workers when starting the worker processes to specify the number of parallel worker processes to initialize. This option should be specified with a positive integer value that is less than or equal to the number of CPU cores of the computer on which the code is being run.

To visualize training results, run the following command:

Tensorboard --logdir=/path/to/desired/logdir

where, instead of the fake argument passed to the option --logdir specified above, you would instead specify the relative or absolute path of the log directory you wish to visualize. This command assumes that the specified log directory contains an events file that can be read by Tensorboard.

To visualize a learned policy for a given environment, run viz.py while passing arguments for the env_id, the policy_file, and the policy_type. env_id should be a string specifying the exact name of an OpenAI Gym environment, and in particular should be for one of the Go, Pong, or Breakout environments (assuming that additional configuration files are not written and additional training is not performed). policy_file should be a string specifying the location of a valid corresponding file of a snapshot of the policy produced at a given iteration of training. policy_type should be a stringspecifying the exact Policy subclass for which we are loading a policy screenshot.
Note that viz.py has not been used extensively throughout this project, and as such is not as well tested as other commands. 

Code for the parallelized evolutionary strategies algorithm was adapted from OpenAI's released evolutionary strategies starter code: https://github.com/openai/evolution-strategies-starter
Code for the A3C algorithm was adapted from OpenAI's released Universe starter agent code for Gym environments: https://github.com/openai/universe-starter-agent
As well, my implementation of A3C and of the combination of ES and A3C adapts some concepts from Denny Britz's A3C implementation: https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
