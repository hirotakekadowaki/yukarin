import argparse
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict

from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
from tb_chainer import SummaryWriter

from utility.chainer_utility import TensorBoardReport
from yukarin.config import create_from_json
from yukarin.dataset import create as create_dataset
from yukarin.model import create
from yukarin.updater import Updater

import chainer
import optuna
from optuna.integration import ChainerPruningExtension

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path', type=Path)
parser.add_argument('output', type=Path)
arguments = parser.parse_args()

config = create_from_json(arguments.config_json_path)
arguments.output.mkdir(exist_ok=True)
config.save_as_json((arguments.output / 'config.json').absolute())

# optuna directory
optuna_dir = Path(arguments.output)/'optuna'
optuna_dir.mkdir(exist_ok=True)
db_file = optuna_dir/'optuna_param.db'
tr_file = optuna_dir/'trainer.npz'
ACC_TH = 0.99
OP_COMP = optuna.structs.TrialState.COMPLETE

def train(trial):
    # model
    if config.train.gpu >= 0:
        cuda.get_device_from_id(config.train.gpu).use()
    predictor, discriminator = create(config.model)
    models = {
        'predictor': predictor,
        'discriminator': discriminator,
    }

    if config.train.pretrained_model is not None:
        serializers.load_npz(str(config.train.pretrained_model), predictor)

    # dataset
    dataset = create_dataset(config.dataset)
    
    if trial.number == 0:
        defbatch = config.train.batchsize
        batchsize = trial.suggest_int('batchsize', defbatch, defbatch)
    else:
        if hasattr(trial, 'state') and trial.state == OP_COMP:
            batchsize = trial.params['batchsize']
        else:
            batchsize = trial.suggest_int('batchsize', 1, 128)
        
    train_iter = MultiprocessIterator(dataset['train'], batchsize)
    
    test_iter = MultiprocessIterator(dataset['test'], batchsize, repeat=False, shuffle=False)
    train_eval_iter = MultiprocessIterator(dataset['train_eval'], batchsize, repeat=False, shuffle=False)
        
    # optimizer
    def create_optimizer(model):
        cp: Dict[str, Any] = copy(config.train.optimizer)
        n = cp.pop('name').lower()

        if n == 'adam':
            if trial.number == 0:
                a = cp.pop('alpha')
                b1 = cp.pop('beta1')
                b2 = cp.pop('beta2')
                alpha = trial.suggest_loguniform('alpha', a, a)
                beta1 = trial.suggest_uniform('beta1', b1, b1)
                beta2 = trial.suggest_uniform('beta2', b2, b2)
                optimizer = optimizers.Adam(alpha, beta1, beta2, eps=10**-7)
            else :
                if hasattr(trial, 'state') and trial.state == OP_COMP:
                    alpha = trial.params['alpha']
                    beta1 = trial.params['beta1']
                    beta2 = trial.params['beta2']
                else:
                    alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-2)
                    beta1 = trial.suggest_uniform('beta1', 0, 1)
                    beta2 = trial.suggest_uniform('beta2', 0, 1)
                    
                optimizer = optimizers.Adam(alpha, beta1, beta2, eps=10**-7)
        elif n == 'sgd':
            optimizer = optimizers.SGD(**cp)
        else:
            raise ValueError(n)
            
        optimizer.setup(model)
        return optimizer
    
    opts = {key: create_optimizer(model) for key, model in models.items()}
    
    # updater
    converter = partial(convert.concat_examples, padding=0)
    updater = Updater(
        loss_config=config.loss,
        predictor=predictor,
        discriminator=discriminator,
        device=config.train.gpu,
        iterator=train_iter,
        optimizer=opts,
        converter=converter,
    )

    # trainer
    trigger_log = (config.train.log_iteration, 'iteration')
    trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
    trigger_snapshot100 = ((int)(config.train.snapshot_iteration*100), 'iteration')
    
    def get_acc_trigger():
        return training.triggers.EarlyStoppingTrigger(
                    check_trigger=trigger_snapshot, 
                    monitor='discriminator/accuracy', 
                    patients=3,
                    mode='max',
                    verbose=False, 
                    max_trigger=trigger_snapshot100)
                    
    def get_loss_trigger():
        return training.triggers.EarlyStoppingTrigger(
                    check_trigger=trigger_snapshot, 
                    monitor='test/predictor/loss', 
                    patients=3,
                    mode='min',
                    verbose=False, 
                    max_trigger=trigger_snapshot100)
    
    trainer = training.Trainer(updater, stop_trigger=get_loss_trigger(), out=arguments.output)
    tb_writer = SummaryWriter(Path(arguments.output))

    ext = extensions.Evaluator(test_iter, models, converter, device=config.train.gpu, eval_func=updater.forward)
    trainer.extend(ext, name='test', trigger=trigger_log)
    ext = extensions.Evaluator(train_eval_iter, models, converter, device=config.train.gpu, eval_func=updater.forward)
    trainer.extend(ext, name='train', trigger=trigger_log)

    #if hasattr(trial, 'state') and trial.state == OP_COMP:
    trainer.extend(extensions.dump_graph('predictor/loss'))
    
    ext = extensions.snapshot_object(predictor, filename='predictor_{.updater.iteration}.npz')
    trainer.extend(ext, name='snapshot', trigger=get_acc_trigger())

    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_snapshot)
    
    # acc
    @training.make_extension(trigger=get_acc_trigger())
    def save_trainer_stop(trainer):
        accuracy = trainer.observation['discriminator/accuracy']
        if accuracy > ACC_TH:
            print('save trainer data!!!')
            chainer.serializers.save_npz(tr_file, trainer, compression=False)
        
    trainer.extend(save_trainer_stop)
    
    # loss
    @training.make_extension(trigger=get_loss_trigger())
    def next_trainer_stop(trainer):
        num = trainer.updater.iteration
        loss = trainer.observation['test/predictor/loss']
        print('losstrigger it:', num, ' / loss:', loss)
        accuracy = trainer.observation['discriminator/accuracy']
        if accuracy > ACC_TH:
            print('save trainer data!!!')
            chainer.serializers.save_npz(tr_file, trainer, compression=False)
            
    trainer.extend(next_trainer_stop)
    
    @training.make_extension(trigger=trigger_snapshot)
    def logacc(trainer):
        num = trainer.updater.iteration
        accuracy = trainer.observation['discriminator/accuracy']
        loss = trainer.observation['discriminator/loss']
        print('it:', num, ' / acc:', accuracy, ' / loss:', loss)
    trainer.extend(logacc)
    
    if not hasattr(trial, 'state'):
        trainer.extend(ChainerPruningExtension(trial, 'discriminator/accuracy', (config.train.snapshot_iteration, 'iteration')))
    
    if tr_file.exists():
        chainer.serializers.load_npz(tr_file, trainer, '', strict=False)
        if hasattr(trial, 'state') and trial.state == OP_COMP:
            alpha = trial.params['alpha']
            beta1 = trial.params['beta1']
            beta2 = trial.params['beta2']
        else:
            alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-2)
            beta1 = trial.suggest_uniform('beta1', 0, 1)
            beta2 = trial.suggest_uniform('beta2', 0, 1)
            
        print(alpha, beta1, beta2)
        opt = trainer.updater.get_optimizer('predictor')
        opt.alpha = alpha
        opt.beta1 = beta1
        opt.beta2 = beta2
        
    trainer.run()
    
    accuracy = trainer.observation['discriminator/accuracy']
    print('optuna next train! / acc:', accuracy)
    return 1.0 - float(accuracy)

if __name__ == '__main__':   
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=config.train.snapshot_iteration)
    dbname = 'sqlite:///'+str(db_file.absolute())
    study = optuna.study.create_study(storage=dbname, pruner=pruner, study_name='yukarin', load_if_exists=True)
    
    study.optimize(train, n_trials=100)
    study.trials_dataframe()[('params', 'alpha')].plot()
    # train
    train(study.best_trial)

