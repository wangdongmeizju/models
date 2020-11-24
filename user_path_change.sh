#!/bin/bash  

echo change to new user path like  ./user_path_change.sh jhon

tmp=`pwd`
tmp1=`echo ${tmp#*/}`
#default_user="`echo ${tmp1%%/*}`"
default_user="inspur"
#new_user=$1
new_user="`echo ${tmp1%%/*}`"

## find all related files 
find  ./* -name "*" | xargs grep  -r $default_user . --exclude=*.sh

## change the default user name 
find -type f -path "./*"  | xargs sed -i 's:/'$default_user'/:/'$new_user'/:g'

## check new uer
find  ./* -name "*" | xargs grep  -r $new_user . --exclude=*.sh
echo $new_user

