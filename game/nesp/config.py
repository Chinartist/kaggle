class CFG:
    wandb=False
    competition='FB3'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=20
    num_workers=0
    #model="Rostlab/prot_bert"
    model="facebook/esm2_t33_650M_UR50D" 
    gradient_checkpointing=False
    scheduler='constant' # ['linear', 'cosine', 'constant']
    batch_scheduler=True
    num_warmup_steps=0
    
    # LEARNING RATE. 
    # Suggested: Prot_bert = 5e-6, ESM2 = 5e-5
    epochs=1
    num_cycles=1.0
    encoder_lr=5e-5
    decoder_lr=5e-5
    batch_size=8
    
    # MODEL INFO - PROT_BERT
    total_layers = 30 
    initial_layers = 5 
    layers_per_block = 16 
    # MODEL INFO - FACEBOOK ESM2
    if 'esm2' in model:
        total_layers = int(model.split('_')[1][1:])
        initial_layers = 2 
        layers_per_block = 15 
        
    # FREEZE
    # Suggested: Prot_bert -8, ESM2 -1
    num_freeze_layers = total_layers - 1
    # NO FREEZE
    #num_freeze_layers = 0
    
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['target']
    seed=42
    n_fold=5
    trn_fold=[0,1,2,3,4]
    train=True
    pca_dim = 64
    device = 2
    context_size = 16