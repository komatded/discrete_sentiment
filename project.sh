#!/bin/bash
docker build -t sentiment_service .
docker stop sentiment01
docker rm sentiment01
docker run -p 8646:8000 -d --name sentiment01 sentiment_service
docker rmi $(docker images -qa -f 'dangling=true')
exit 0