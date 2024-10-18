dataset=$1

#convert
python convert.py -s data/$dataset

#3dgs
python train.py 

# render
python render.py -s data/$dataset -m output/${dataset}_0 --feature_level 0 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_1 --feature_level 1 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_2 --feature_level 2 --include_feature;
python render.py -s data/$dataset -m output/${dataset}_3 --feature_level 3 --include_feature;