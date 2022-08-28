Whats-The-Food-Cloud-Api
==
Works Steps to build and deploy in Cloud Run
--
1. First, we clone the gihub repository containing the REST API and the required files to the google cloud platform.
2. Next, we create a container image from the dockerfile that we cloned earlier.

docker build -t foodapp:0.5 .
Then we push to the container registry in GCP
docker tag foodapp:0.5 gcr.io/braided-period-345503/foodapp:0.5
docker push gcr.io/braided-period-345503/foodapp:0.5
Finally, we just deploy to cloud run
gcloud run deploy foodapp \
--image=gcr.io/braided-period-345503/cc-cloud-api \
--region=asia-southeast2 \
--project=braided-period-345503 \
 && gcloud run services update-traffic foodapp --to-latest
