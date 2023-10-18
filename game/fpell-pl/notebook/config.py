from types import SimpleNamespace
config = SimpleNamespace(**{})

config.train='/home/wangjingqi/input/dataset/fpell/train/train.csv'
config.test='/home/wangjingqi/input/dataset/fpell/test.csv'
config.submit='/home/wangjingqi/input/dataset/fpell/sample_submission.csv'
config.ck='/home/wangjingqi/input/ck/fpell'
config.log='/home/wangjingqi/fpell-pl/log'

config.precision=16
config.patience=10
config.nfolds=5
config.used_folds=[0, 1, 2 ,3 ,4]
#jf
config.bs=4
config.accumulate_grad_batches=None
config.epochs=5
config.val_check_interval = 1.0
config.device_ids=[1]
config.head_lr=10e-5  

config.freeze=None

config.bert=["microsoft/deberta-large"][0]
config.prefix=f'lr1'

config.llrd=0.4
config.wd=0
config.llrd_interval = 1
config.reinit_layers=1
config.layer_start=-1
config.max_len=1024
config.seed=2022
config.model_fname=f'{config.bert.split("/")[-1]}_{config.prefix}-{config.max_len}-{config.epochs}-{config.seed}'
print(config)