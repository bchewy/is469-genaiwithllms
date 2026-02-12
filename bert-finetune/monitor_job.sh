#!/bin/bash
# Live monitor for BERT fine-tuning job
# Usage: ./monitor_job.sh [job_id]
#   If no job_id given, auto-detects from myqueue

JOB_ID=${1:-$(squeue --me --noheader -o "%i" 2>/dev/null | head -1)}
if [ -z "$JOB_ID" ]; then
    echo "No running jobs found. Check with: myqueue"
    exit 1
fi

OUTFILE=$(ls -t ~/bert-yelp-finetune.${JOB_ID}.out 2>/dev/null)
if [ -z "$OUTFILE" ]; then
    # fallback: find any matching output
    OUTFILE=$(ls -t ~/bert-yelp-finetune.*.out 2>/dev/null | head -1)
fi

if [ -z "$OUTFILE" ]; then
    echo "No output file found for job $JOB_ID"
    exit 1
fi

echo "Monitoring job $JOB_ID — output: $OUTFILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "====== BERT Yelp Fine-tuning Monitor ======"
    echo "Job ID: $JOB_ID | $(date '+%H:%M:%S')"

    # Job state
    STATE=$(squeue --job=$JOB_ID --noheader -o "%T %M %l %N" 2>/dev/null)
    if [ -n "$STATE" ]; then
        echo "Status: $STATE"
    else
        echo "Status: COMPLETED/NOT RUNNING"
    fi
    echo ""

    # GPU line
    grep -m1 "NVIDIA" "$OUTFILE" 2>/dev/null

    # Latest training progress (last progress bar line)
    PROGRESS=$(grep -oP '\d+/60939.*?it/s\]' "$OUTFILE" 2>/dev/null | tail -1)
    if [ -n "$PROGRESS" ]; then
        echo ""
        echo "Training: $PROGRESS"
        # Calculate percentage
        STEP=$(echo "$PROGRESS" | grep -oP '^\d+')
        PCT=$(echo "scale=1; $STEP * 100 / 60939" | bc 2>/dev/null)
        echo "Progress: ${PCT}% ($STEP / 60939 steps)"
    fi

    # Latest eval results
    EVAL=$(grep "'eval_accuracy'" "$OUTFILE" 2>/dev/null | tail -1)
    if [ -n "$EVAL" ]; then
        echo ""
        echo "Latest eval: $EVAL"
    fi

    # Latest loss logging
    LOSS=$(grep "'loss'" "$OUTFILE" 2>/dev/null | tail -1)
    if [ -n "$LOSS" ]; then
        echo "Latest loss: $LOSS"
    fi

    # Check for final results
    FINAL=$(grep "FINAL ACCURACY" "$OUTFILE" 2>/dev/null)
    if [ -n "$FINAL" ]; then
        echo ""
        echo "========================================="
        echo "$FINAL"
        echo "========================================="
        echo ""
        echo "Job finished! Full results:"
        grep -A2 "Test Accuracy" "$OUTFILE" 2>/dev/null | tail -3
        echo ""
        grep -A20 "Classification Report" "$OUTFILE" 2>/dev/null | tail -15
        break
    fi

    sleep 30
done
