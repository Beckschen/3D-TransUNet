import pickle
# './nnUNetPlansv2.1_plans_3D_p96224224.pkl'
picke_path = '../../data/nnUNet_preprocessed/Task801_WORD/splits_final.pkl'
a = pickle.load(open(picke_path, 'rb'))
print(a[0]['train'].keys())