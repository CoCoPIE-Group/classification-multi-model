DEFAULT_PRUNING_CONFIG = {
    "sp_store_weights": None,
    "sp_lars": False,
    "sp_lars_trust_coef": 0.001,
    "sp_backbone": False,
    "sp_retrain": False,
    "sp_admm": False,
    "sp_admm_multi": False,
    "sp_retrain_multi": False,
    "sp_config_file": None,
    "sp_subset_progressive": False,
    "sp_admm_fixed_params": False,
    "sp_no_harden": False,
    "nv_sparse": False,
    "sp_load_prune_params": None,
    "sp_store_prune_params": None,
    "generate_rand_seq_gap_yaml": False,
    "sp_admm_update_epoch": 5,
    "sp_admm_update_batch": None,
    "sp_admm_rho": 0.001,
    "sparsity_type": "block_punched",
    "sp_admm_lr": 0.01,
    "admm_debug": False,
    "sp_global_weight_sparsity": -1,
    "sp_prune_threshold": -1.0,
    "sp_block_irregular_sparsity": "(0,0)",
    "sp_block_permute_multiplier": 2,
    "sp_admm_block": "(8,4)",
    "sp_admm_buckets_num": 16,
    "sp_admm_elem_per_row": 1,
    "sp_admm_tile": None,
    "sp_admm_select_number": 4,
    "sp_admm_pattern_row_sub": 1,
    "sp_admm_pattern_col_sub": 4,
    "sp_admm_data_format": None,
    "sp_admm_do_not_permute_conv": False,
    "sp_gs_output_v": None,
    "sp_gs_output_ptr": None,
    "sp_load_frozen_weights": None,
    "retrain_mask_pattern": "weight",
    "sp_update_init_method": "weight",
    "sp_mask_update_freq": 10,
    "retrain_mask_sparsity": -1.0,
    "retrain_mask_seed": None,
    "sp_prune_before_retrain": False,
    "output_compressed_format": False,
    "sp_grad_update": False,
    "sp_grad_decay": 0.98,
    "sp_grad_restore_threshold": -1,
    "sp_global_magnitude": False,
    "sp_pre_defined_mask_dir": None,
    "sp_prune_ratios": 0,
}
