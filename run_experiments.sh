for i in `seq 0 4`;do
  python main.py --cuda=1 --i=$i --config="$1"
done
