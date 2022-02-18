# DisMLN

# Prepare dataset

```bash
sh dset_gen/pipline.sh # change the split
```
or link to dir
```
ln -s /data/leuven/335/vsc33595/dataset/mln_v1 data
cd dset_gen
ln -s /scratch/leuven/335/vsc33595/dset_cache generated
```

# Train model

```bash
sh exps/pointnet2_bert/run_cached.sh
```

