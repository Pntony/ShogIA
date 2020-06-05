import sys

colab_folder = "/content/drive/My Drive/Colab Notebooks/SnnAI"
sys.path.insert(0, colab_folder)





class Configuration:
# Contains all the values of the parameters of the program.
# The values indicated in commentary are those used in Alpha(Go) Zero.
    
    def __init__(self):

        self.load = False

        # Files
        self.folder = "/content/drive/My Drive/Colab Notebooks/checkpoints/session1/"
        self.run_manager_file_name = self.folder + "runManager.json"
        self.nnet_file_name = self.folder + "nnet.pt"
        self.buffer_file_name = self.folder + "buffer.json"
        self.histories_file_name = self.folder + "histories.npy"
        self.state_checkpoints_file_name = self.folder + "stateCheckpoints.npy"

        # Hyperparameters
        self.num_kernels = 256 # 64 # 256
        self.num_res_blocks = 19 # 14 # 19
        self.lr = 0.01 # 0.01 # 0.2 -> 0.02 -> 0.002 -> 0.0002
        self.weight_decay = 1e-4 # 1e-4
        self.alpha_dir = 0.15 # 0.15
        self.eps_dir = 0.25 # 0.25
        self.eval_move = 1

        # Number iterations
        self.num_iter = 700 # 700 (= 700,000 / 1,000 ; 700,000 training steps / gradient descent steps)
        self.games_per_iter = 25000 # 25,000
        self.epochs_per_iter = 8000 # 1e3
        self.games_save_interval = 5000
        self.epochs_save_interval = 1000000
        self.max_moves = 512 # 512
        self.num_eval_games = 1000

        # Data sizes
        self.window_size = int(1e5) # 1e6
        self.batch_size = 1024 # 1000 # 4096
        self.parallel_games = 25000
        self.state_checkpoint_interval = 50

        # For monitoring
        self.game_code_num_moves = 5





if __name__ == "__main__":

    import trainNNet

    c = Configuration()
    nnet_trainer = trainNNet.NNetTrainer(c)
    nnet_trainer.train_nnet()