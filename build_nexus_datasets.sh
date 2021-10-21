tar -xf Nexus5_Kernel_BIOTracer_traces.tar.gz 
for i in WorkSpace_nexus5/Trace_files/log{126,176,225,245}*.txt
do 
	cat $i | cut -f 9,15 | sort -n | awk '{if (NR > 1 && last < $1) {print 5000 * ($1 - last)}; last=((NR > 1 && last > $2)?last:$2)}' | awk 'NR % 5 == 0 {print $1}' | awk '$1 < 20 {print $1}' > data/nexus_`basename $i`
done
