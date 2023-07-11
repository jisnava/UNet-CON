class config:
#     train_path = "/home/p170107cs/RaptorX_dataset/pdb25-6767-train.release.contactFeatures.pkl"
#     valid_path = "/home/p170107cs/RaptorX_dataset/pdb25-6767-valid.release.contactFeatures.pkl"
#     test_path = "/home/p170107cs/RaptorX_dataset/pdb25-test-500.release.contactFeatures.pkl"
    train_path = "pdb25-6767-train.release.contactFeatures.pkl"
    valid_path = "pdb25-6767-valid.release.contactFeatures.pkl"
    test_path = "pdb25-test-500.release.contactFeatures.pkl"
    lr = 3e-2
    k_splits = 6
    max_len = 256
    pad_size = 10
    num_blocks = 16
    expected_n_channels = 57
    protein_len_consider = 50 //Minimum length
    weight_decay_1 = 0.001
    weight_decay_2 = 0.0
    epochs = 100
    best_val_min = -1

# lr = 3e-2   			         #Learning rate
# k_splits = 6 			         #K-fold validation with fold=6
# max_len = 256 		         #Maximum length of protein sequence under consideration
# expected_n_channels = 57   	 #Number of channels
# protein_len_consider = 50 	 #Minimum length of protein sequence
# weight_decay_1 = 0.001   	     #Weight decay from model layers
# weight_decay_2 = 0.0		     #Weight decay from model layers
# epochs = 100			         #Number of epochs
# best_val_min = -1		         #Initial validation score
# pad_size = 10                  #For convolutions

