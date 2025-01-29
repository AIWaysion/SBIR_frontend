#!/bin/bash

LOGFILE="mylogfile.log"
MAXSIZE=$((100 * 1024 * 1024)) # 100MB

while true; do
    if [ -f "$LOGFILE" ] && [ $(stat -c%s "$LOGFILE") -gt $MAXSIZE ]; then
        > "$LOGFILE"
    fi
    pkill -f "python run.py"
    python run.py >> "$LOGFILE" 2>&1 &
    sleep 1800
done
