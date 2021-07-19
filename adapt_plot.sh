#!/bin/bash

echo "plots PSK dataset 10000 instances uniform between 1 and 4"

./main.py -l results/psk-4-10000-rhos.json  -k rhos            -o results/psk-4-10000-rhos            data/psk_4_10000.txt
./main.py -l results/psk-4-10000-rhos.json  -k best            -o results/psk-4-10000-best            data/psk_4_10000.txt
./main.py -l results/psk-4-10000-rhos.json  -k bestmult        -o results/psk-4-10000-bestmult        data/psk_4_10000.txt
./main.py -l results/psk-4-10000-rhos.json  -k ouralg_manyrhos -o results/psk-4-10000-ouralg_manyrhos data/psk_4_10000.txt

echo "plots PSK dataset 10000 instances uniform between 1 and 8"

./main.py -l results/psk-8-10000-bestmult.json -k bestmult -o results/psk-8-10000-bestmult data/psk_8_10000.txt
