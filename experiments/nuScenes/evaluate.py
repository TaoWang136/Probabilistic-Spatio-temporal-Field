import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
parser.add_argument("--site", type=str, default=None)



args = parser.parse_args()




def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            ############### MOST LIKELY Z ###############

            print("-- Evaluating GMM Z Mode (Most Likely)")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)
                #print('测试-1',timesteps)

                predictions = eval_stg.predict(
                                               scene,
                                               timesteps,
                                               ph,
                                               site=args.site,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=True,
                                               save=True)  # This will trigger grid sampling
                #print('测试1')

                pre,tru,his = evaluation.compute_batch_statistics_test(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False,
                                                                       kde=False)


            ############## FULL ###############
            print("-- Evaluating Full")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               site=args.site,
                                               num_samples=300,
                                               min_future_timesteps=8,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False,
                                               save=False)

                if not predictions:
                    continue
                prediction_dict, _, _ = utils.prediction_output_to_trajectories(predictions,
                                                                                scene.dt,
                                                                                max_hl,
                                                                                ph,
                                                                                prune_ph_to_future=False)
                pre_full,tru,his = evaluation.compute_batch_statistics_test(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False)                                                                                                         
                save_path_pre = f'/home/wangtao/下载/Trajectron-ped-veh/Data_ana/Data_gen/{args.site}/{args.site}_pre_full_{len(scene.nodes)}_{len(timesteps)}.pkl'
                os.makedirs(os.path.dirname(save_path_pre), exist_ok=True)
                with open(save_path_pre, 'wb') as f:
                    pickle.dump(pre_full, f)
                                        
                save_path_tru = f'/home/wangtao/下载/Trajectron-ped-veh/Data_ana/Data_gen/{args.site}/{args.site}_tru_{len(scene.nodes)}_{len(timesteps)}.pkl'
                os.makedirs(os.path.dirname(save_path_tru), exist_ok=True)
                with open(save_path_tru, 'wb') as f:
                    pickle.dump(tru, f)   
                                   
                save_path_his= f'/home/wangtao/下载/Trajectron-ped-veh/Data_ana/Data_gen/{args.site}/{args.site}_his_{len(scene.nodes)}_{len(timesteps)}.pkl'
                os.makedirs(os.path.dirname(save_path_his), exist_ok=True)
                with open(save_path_his, 'wb') as f:
                    pickle.dump(his, f)   
                      
     