from argparse import ArgumentParser
def make_args():
	parser = ArgumentParser()
	parser.add_argument("-trainpath", help="Path to train data",
    default = "/home/sahil/work/deepgttGIT/data/trainpath")

	parser.add_argument("-validpath", help="Path to validate data",
	    default = "/home/sahil/work/deepgttGIT/data/validpath")

	# parser.add_argument("-dim_u", type=int, default=200,
	    # help="The dimension of embedding u")

	# parser.add_argument("-dim_rho", type=int, default=256,
	    # help="The dimension of rho (road representation)")

	parser.add_argument("-dim_pi", type=int, default=1000,
	    help="The number of proxy destinations (K)")

	parser.add_argument("-dim_c", type=int, default=400,
	    help="The dimension of c (traffic state representation)")

	parser.add_argument("-dim_h", type=int, default=256,
	    help="The dimension of pi (proxy representation)")

	parser.add_argument("-dim_n_r", type=int, default=128,
	    help="The dimension of pi (proxy representation)")

	parser.add_argument("-dim_n_x", type=int, default=128,
	    help="The dimension of pi (proxy representation)")

	parser.add_argument("-dropout", type=float, default=0.2,
	    help="The dropout probability")

	parser.add_argument("-batch_size", type=int, default=150,
	    help="The batch size")

	parser.add_argument("-num_epoch", type=int, default=100,
	    help="The number of epoch")

	parser.add_argument("-max_grad_norm", type=float, default=0.1,
	    help="The maximum gradient norm")

	parser.add_argument("-lr", type=float, default=0.001,
	    help="Learning rate")

	parser.add_argument("-lr_decay", type=float, default=0.2,
	    help="Learning rate decay")

	parser.add_argument("-kl_decay", type=float, default=0,
	    help="KL Divergence decay")

	parser.add_argument("-print_freq", type=int, default=1000,
	    help="Print frequency")

	parser.add_argument("-use_cuda", type=bool, default=True)

	parser.add_argument("-use_selu", action="store_true")

	parser.add_argument("-random_emit", action="store_true")

	parser.add_argument("-temp", type=float, default=0.8)

	parser.add_argument("-hard", type=bool, default=False)

	parser.add_argument("-use_cache", type=bool, default=False)
	args = parser.parse_args()
	return args
