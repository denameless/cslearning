def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')
    parser.add_argument("--seed", default=11037, type=int, help="random seed")
    parser.add_argument("--use_cuda", default=True, type=bool, help="whether to use cuda if available")

    # Q-Network
    parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of Q-network for CartPole")

    # Optimizer
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for Adam optimizer")
    parser.add_argument("--grad_norm_clip", default=10.0, type=float, help="gradient norm clipping")

    # DQN specific
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--buffer_size", default=10000, type=int, help="replay buffer size")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training")
    # 'd' from the algorithm image (目标网络更新间隔 d)
    parser.add_argument("--target_update_interval_d", default=100, type=int, help="target network update interval 'd' (steps within an episode, as per algorithm image)")
    parser.add_argument("--n_frames", default=int(50000), type=int, help="total number of training frames/steps")

    # Epsilon-greedy exploration
    parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial epsilon value for exploration")
    parser.add_argument("--epsilon_end", default=0.01, type=float, help="final epsilon value for exploration")
    parser.add_argument("--epsilon_decay_frames", default=10000, type=int, help="frames over which to decay epsilon (based on total environment steps)")

    parser.add_argument("--test", default=False, type=bool, help="test mode (no exploration, for make_action)")
    return parser



def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name') # PG might use different envs too

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int) # PG might use smaller nets
    parser.add_argument("--lr", default=0.01, type=float) # PG often uses different LR
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int) # PG might train faster/slower

    return parser
