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
