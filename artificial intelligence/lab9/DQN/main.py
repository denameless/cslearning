import argparse
import gym
from argument import dqn_arguments, pg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")

    # 使用 type=bool 需要命令行传递 'True' 或 'False'
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=False, type=bool, help='whether train DQN')

    print("Initial parser arguments defined.")

    temp_args, unknown_args = parser.parse_known_args() # 获取未知参数以备检查
    print(f"--- temp_args from parse_known_args: {temp_args}") 
    print(f"--- temp_args.train_dqn: {temp_args.train_dqn}, type: {type(temp_args.train_dqn)}")
    if unknown_args:
        print(f"--- Unknown arguments after first parse: {unknown_args}") 

    if temp_args.train_dqn is True:
        print("--- Loading DQN arguments because temp_args.train_dqn is True ---") 
        parser = dqn_arguments(parser)
    elif temp_args.train_pg is True:
        print("--- Loading PG arguments because temp_args.train_pg is True ---") 
        parser = pg_arguments(parser)
    else:
        print(f"--- Neither DQN nor PG arguments loaded. train_dqn: {temp_args.train_dqn}, train_pg: {temp_args.train_pg} ---") 

    args = parser.parse_args()
    print(f"--- Final parsed args: {args} ---") 
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()


if __name__ == '__main__':
    args = parse()
    run(args)
