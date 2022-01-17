import sys
from subprocess import check_call

PYTHON = sys.executable

def launch_training_job(model_dir, model, restore):
    if restore is not None:
        cmd = "{python} train.py --model_dir={model_dir} --model_type={model_type} --restore={restore}".format(python=PYTHON, model_dir=model_dir, model_type=model, restore=restore)
    else:
        cmd = "{python} train.py --model_dir={model_dir} --model_type={model_type}".format(python=PYTHON, model_dir=model_dir, model_type=model)
    check_call(cmd, shell=True)

if __name__ == '__main__':
    # BASE_PATH = './experiments/satori/'
    BASE_PATH = './experiments/'
    
    
    folders = [   
                {'dataset':'satori', 'model_dir':'hop2_model', 'model':'base_model', 'restore': None},
                {'dataset':'wikidata','model_dir':'hop2_model', 'model':'base_model', 'restore': None},
                {'dataset':'music','model_dir':'hop2_model', 'model':'base_model', 'restore': None},
                
                #{'dataset':'satori','model_dir':'balance_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'balance_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'balance_hop2_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'no_overlap_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'no_overlap_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'no_overlap_hop2_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'dynamic_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'dynamic_hop2_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'dynamic_hop2_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'dynamic_hop1_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'dynamic_hop1_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'dynamic_hop1_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'dynamic_hop3_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'dynamic_hop3_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'dynamic_hop3_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'dynamic_hop4_model', 'model':'base_model', 'restore': None},
                #{'dataset':'wikidata','model_dir':'dynamic_hop4_model', 'model':'base_model', 'restore': None},
                #{'dataset':'music','model_dir':'dynamic_hop4_model', 'model':'base_model', 'restore': None},

                #{'dataset':'satori','model_dir':'dynamic_hop0_model', 'model':'HOP0_CKAN', 'restore': None},
                #{'dataset':'wikidata','model_dir':'dynamic_hop0_model', 'model':'HOP0_CKAN', 'restore': None},
                #{'dataset':'music','model_dir':'dynamic_hop0_model', 'model':'HOP0_CKAN', 'restore': None},
                
    ]
    
    for folder in folders:
        model_dir = BASE_PATH + folder['dataset'] +'/' +folder['model_dir']
        
        launch_training_job(model_dir, folder['model'], folder['restore'])
    