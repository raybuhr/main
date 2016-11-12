#!/bin/bash

# Submit script for assignment1
#
# BEFORE YOU RUN THIS: check Piazza for the invite link, which will create a
# private submission repo linked to your GitHub username.
#
# Usage: ./submit.sh my_github_username

GITHUB_USERNAME=${1:-"$USER"}

set -e

REPO_NAME="datasci-w266/assignment1-$GITHUB_USERNAME"

echo "Select preferred GitHub access protocol:"
echo "HTTPS is default, but SSH may be needed if you use two-factor auth."
select mode in "HTTPS" "SSH" "(cancel)"; do
  case $mode in
    "HTTPS" ) REMOTE_URL="https://github.com/$REPO_NAME.git"; break;;
    "SSH" ) REMOTE_URL="git@github.com:$REPO_NAME.git"; break;;
    "(cancel)" ) exit;;
  esac
done

# Set up git remote
REMOTE_ALIAS="assignment1-submit"
echo "== Pushing to submission repo $REPO_NAME"
echo "== Latest commit: $(git rev-parse HEAD)"
echo "== Check submission status at https://github.com/$REPO_NAME"
if [[ $(git remote | grep "$REMOTE_ALIAS") ]]; then
  git remote -v remove "$REMOTE_ALIAS"
fi
git remote -v add "$REMOTE_ALIAS" "${REMOTE_URL}"
git push "$REMOTE_ALIAS" master

# Verify submission succeeded
git fetch "$REMOTE_ALIAS"
if [[ $(git rev-parse HEAD) == $(git ls-remote "$REMOTE_ALIAS" HEAD | cut -f1) ]]; then
  echo "=== Submission successful! ==="
else
  echo "=== ERROR: Submission failed. Manually push this repo to $REPO_NAME, or
  contact the course staff for help."
fi
