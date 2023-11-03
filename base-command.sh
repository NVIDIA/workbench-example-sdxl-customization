#!/bin/bash

# *** AUTO-GENERATED FILE - DO NOT EDIT ****
# This script is required when running projects in NGC Base Command from Workbench
# It is safe to delete if you do not plan to run this project in NGC Base Command

if [ -z "$NVWB_BASE_COMMAND" ]
then
    echo "base-command.sh should only be used while executing in Base Command."
	exit 1
fi

# Wipe workspace contents prior to clone
echo "Erasing project workspace mounted at /project"
cd /project
rm -rf ..?* .[!.]* *
ls -la /project

# Clone the repo into the workspace
echo "Cloning https://${NVWB_PROJECT_URL} into /project"
git clone https://${NVWB_GIT_USERNAME}:${NVWB_GIT_PASSWORD}@${NVWB_PROJECT_URL} /project

# Run the command
echo "running $@"
cd /project$NVWB_SCRIPT_DIR
$@
