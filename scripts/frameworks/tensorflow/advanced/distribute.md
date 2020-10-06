# `tf.distribute`(2.3)

`tf.distribute.Strategy`

`tf.distribute.MirroredStrategy`
`tf.distribute.experimental.MultiWorkerMirroredStrategy`
`tf.distribute.TPUStrategy`
`tf.distribute.experimental.CentralStorageStrategy`
`tf.distribute.experimental.ParameterServerStrategy`


## Types of Strategy

### MirroredStrategy

```py
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
```

### TPUStrategy

```py
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=tpu_address)
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)
```

### MultiWorkerMirroredStrategy

```py
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# tf.distribute.experimental.CollectiveCommunication.RING
# tf.distribute.experimental.CollectiveCommunication.NCCL
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.AUTO) 
```

### CentralStorageStrategy

```py
central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
```
