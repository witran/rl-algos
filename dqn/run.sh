python monitor_bench.py queue
python monitor_bench.py shm
~/code/labs/gprof2dot/gprof2dot.py -f pstats monitor_queue_0.cprofile | dot -Tpng -o callgraph_queue.png
~/code/labs/gprof2dot/gprof2dot.py -f pstats monitor_shm_0.cprofile | dot -Tpng -o callgraph_shm.png
open callgraph_queue.png
open callgraph_shm.png
