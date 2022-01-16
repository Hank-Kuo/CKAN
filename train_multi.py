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
    BASE_PATH = './experiments/satori/'
    
    folders = [   
                {'model_dir':'base_model', 'model':'base_model', 'restore': None},
    ]
    
    for folder in folders:
        model_dir = BASE_PATH + folder['model_dir']
        launch_training_job(model_dir, folder['model'], folder['restore'])
    