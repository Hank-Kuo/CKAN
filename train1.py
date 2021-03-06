import os
import argparse
import json
from tqdm import tqdm
import logging

import model.net as net
from evaluate import evaluation
import model.data_loader as data_loader
import utils.load_data_dynamic as load_data
import utils.utils as utils
import utils.tensorboard as tensorboard

import numpy as np
import torch
from torch.utils import data as torch_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="./experiments/satori/base_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_type", default="base_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--restore", default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")


def get_model(params, model_type):
    model = {
        'base_model': net.CKAN(params),
        'hop0_model': net.HOP0_CKAN(params),
    }

    return model[model_type]
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return int(obj)


def main():
    args = parser.parse_args()
    
    # torch setting
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    params_path = os.path.join(args.model_dir, 'params.json')
    user_ripple_path = os.path.join(args.model_dir, 'user_set.json')
    item_ripple_path = os.path.join(args.model_dir, 'item_set.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    test_best_json_path = os.path.join(args.model_dir, "metrics_test_best_weights.json")

    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # load dataset
    print("===> Loading datasets")
    train_data, test_data, n_entity, n_relation, max_user_history_item, kg, user_init_entity_set, item_init_entity_set, user_triple_sets, item_triple_sets = load_data.load_data(params)
    params.n_entity = n_entity
    params.n_relation = n_relation

    # data loader
    train_set = data_loader.Dataset(params, train_data, user_triple_sets, item_triple_sets )
    test_set = data_loader.Dataset(params, test_data, user_triple_sets, item_triple_sets )
    train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size, drop_last=False, shuffle=True)
    test_generator = torch_data.DataLoader(test_set, batch_size=params.batch_size, drop_last=False)
    
    # model
    print("===> Building model")
    model = get_model(params, args.model_type)

    model = model.to(params.device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = params.learning_rate,
        weight_decay = params.l2_weight,
    )
    tb = tensorboard.Tensorboard(args.model_dir, False)
    writer = tb.create_writer()
    
    start_epoch_id, step, best_score = 1, 0 , 0.0
    patient, temp = 0, 0

    if args.restore is not None:
        print('Restore checkpoint...')
        start_epoch_id, step, best_score = utils.load_checkpoint(checkpoint_dir, model, optimizer)

    logging.info("Number of Entity: {}, Number of Relation: {}, User History item: {}".format(n_entity, n_relation, max_user_history_item))
    logging.info("Training Dataset: {}, Test Dataset: {}".format(len(train_set), len(test_set)))
    logging.info("MODEL NAME: {}".format(model.__class__.__name__))
    logging.info("SAMPLER NAME: {}".format(load_data.__name__))
    logging.info("===> Starting training ...")
    print(model)

    # Train
    for epoch_id in range(start_epoch_id, params.epochs + 1):
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        with tqdm(total=len(train_generator)) as t:
            for users, items, labels, user_triplets, item_triplets in train_generator:
                items = items.to(params.device)
                labels = labels.to(params.device)
                return_dict, _ = model(items, labels, user_triplets, item_triplets)
                
                loss = return_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_impacting_samples_count += loss.item()
                samples_count += items.size()[0]
                step += 1

                t.set_postfix(loss = loss_impacting_samples_count / samples_count * 100)
                t.update()
                
                writer.add_scalar('Loss/base_loss', return_dict["loss"].data.cpu().numpy(), global_step=step)
        
        # validation
        if epoch_id % params.valid_every == 0:
            model.eval()

            test_metrics = evaluation(params, model, test_generator)
            logging.info('- Eval: test auc: %.4f  acc: %.4f  f1: %.4f'% (test_metrics['auc'], test_metrics['acc'], test_metrics['f1']))
            
            writer.add_scalar('Accuracy/test/AUC', test_metrics['auc'] , global_step=epoch_id)
            writer.add_scalar('Accuracy/test/ACC', test_metrics['acc'] , global_step=epoch_id)
            writer.add_scalar('Accuracy/test/F1', test_metrics['f1'] , global_step=epoch_id)
            
            score = test_metrics['auc']
            if score > best_score:
                best_score = score   
                test_metrics['epoch'] = epoch_id   
                       
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, step, best_score)
                utils.save_dict_to_json(test_metrics, test_best_json_path)

                with open(user_ripple_path, 'w') as f:
                    json.dump(user_triple_sets, f, cls=NpEncoder)
                with open(item_ripple_path, 'w') as f:
                    json.dump(item_triple_sets, f, cls=NpEncoder)

            if temp < score:
                temp = score
            else:
                patient +=1 

        # update ripple set 
        if epoch_id %  params.ripple_update == 0 or patient >= params.patience:    
            user_triple_sets = load_data.kg_propagation(params, kg, user_init_entity_set, max_user_history_item, True)
            item_triple_sets = load_data.kg_propagation(params, kg, item_init_entity_set,  max_user_history_item , True)
            
            train_set = data_loader.Dataset(params, train_data, user_triple_sets, item_triple_sets )
            test_set = data_loader.Dataset(params, test_data, user_triple_sets, item_triple_sets )
            train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size, drop_last=False, shuffle=True)
            test_generator = torch_data.DataLoader(test_set, batch_size=params.batch_size, drop_last=False)

            patient = 0 
            temp = 0
                    
    tb.finalize()

if __name__ == '__main__':
    main()