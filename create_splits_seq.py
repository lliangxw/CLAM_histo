import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np


def add_labels_and_reorder_split_csv(
    split_csv_path: str,
    source_csv_path: str,
    slide_col: str = "slide_id",
    label_col: str = "label",
    out_csv_path: str | None = None,
):
    # Load source CSV and build slide_id -> label mapping
    src = pd.read_csv(source_csv_path)

    if slide_col not in src.columns:
        raise ValueError(f"Column '{slide_col}' not found in source CSV. Available columns: {list(src.columns)}")

    if label_col not in src.columns:
        raise ValueError(f"Column '{label_col}' not found in source CSV. Available columns: {list(src.columns)}")

    id2label = dict(zip(src[slide_col].astype(str), src[label_col]))

    # Load split CSV
    sp = pd.read_csv(split_csv_path)

    # Create label columns
    for c in ["train", "val", "test"]:
        if c in sp.columns:
            sp[c + "_label"] = sp[c].map(id2label).astype('Int64')
            

    # Reorder columns as: train | train_label | val | val_label | test | test_label
    desired_order = ["train", "train_label", "val", "val_label", "test", "test_label"]
    existing_desired = [c for c in desired_order if c in sp.columns]
    remaining = [c for c in sp.columns if c not in existing_desired]

    sp = sp[existing_desired + remaining]

    sp.insert(0, "", range(len(sp)))

    # Save
    if out_csv_path is None:
        out_csv_path = split_csv_path

    sp.to_csv(out_csv_path, index=False)

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_CCA_vs_HCC', 'task_2_CCA_vs_HCC_vs_TM', 'task_3_CCA_vs_HCC_TM', 'task_4_CCA_TM_vs_HCC', 'task_5_CCA_vs_TM', 'task_6_HCC_vs_TM'], required=True,)
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.0,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_CCA_vs_HCC':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CCA_vs_HCC_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_CCA_vs_HCC_vs_TM':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CCA_vs_HCC_vs_TM_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1, '2':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_3_CCA_vs_HCC_TM':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CCA_vs_HCC_TM_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_4_CCA_TM_vs_HCC':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CCA_TM_vs_HCC_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_5_CCA_vs_TM':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/CCA_vs_TM_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_6_HCC_vs_TM':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/HCC_vs_TM_256.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

### For CLAM
# if __name__ == '__main__':
#     if args.label_frac > 0:
#         label_fracs = [args.label_frac]
#     else:
#         label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
#     for lf in label_fracs:
#         split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
#         os.makedirs(split_dir, exist_ok=True)
#         dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
#         for i in range(args.k):
#             dataset.set_splits()
#             descriptor_df = dataset.test_split_gen(return_descriptor=True)
#             splits = dataset.return_splits(from_id=True)
#             save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))

#             #### Added for TransMIL
#             # source_csv = "dataset_csv/CCA_vs_HCC_TM_256.csv"

#             # split_csv = os.path.join(split_dir, f"fold{i}.csv")

#             # save_splits(splits, ["train", "val", "test"], split_csv)

#             # add_labels_and_reorder_split_csv(
#             #     split_csv_path=split_csv,
#             #     source_csv_path=source_csv,
#             #     slide_col="slide_id",
#             #     label_col="label",
#             #     out_csv_path=split_csv
#             # )

#             ######

#             save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
#             descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))

### For TransMIL
# # 
if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = '/home/local-admin/Documents/projects/TransMIL/dataset_csv/'+ str(args.task)
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))

            #### Added for TransMIL
            source_csv = "dataset_csv/HCC_vs_TM_256.csv"

            split_csv = os.path.join(split_dir, f"fold{i}.csv")

            save_splits(splits, ["train", "val", "test"], split_csv)

            add_labels_and_reorder_split_csv(
                split_csv_path=split_csv,
                source_csv_path=source_csv,
                slide_col="slide_id",
                label_col="label",
                out_csv_path=split_csv
            )

  




