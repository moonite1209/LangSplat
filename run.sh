dataset=$1

# convert
python convert.py -s data/$dataset

# 3dgs
python train.py -s data/$dataset -m output/$dataset

# autoencoder
python autoencoder/train.py -n ${dataset} -s data/${dataset}
python autoencoder/test.py -n ${dataset} -s data/${dataset}

# langsplat
python train.py -s data/${dataset} -m output/${dataset} --start_checkpoint output/3dgs/$dataset/chkpnt30000.pth --feature_level 0;
python train.py -s data/${dataset} -m output/${dataset} --start_checkpoint output/3dgs/$dataset/chkpnt30000.pth --feature_level 1;
python train.py -s data/${dataset} -m output/${dataset} --start_checkpoint output/3dgs/$dataset/chkpnt30000.pth --feature_level 2;
python train.py -s data/${dataset} -m output/${dataset} --start_checkpoint output/3dgs/$dataset/chkpnt30000.pth --feature_level 3;

# render
python render.py -s data/$dataset -m output/${dataset}_0 --feature_level 0 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_1 --feature_level 1 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_2 --feature_level 2 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_3 --feature_level 3 --include_feature;