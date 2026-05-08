# Dynamic Sparse Training Demo

## Run locally

1. From this folder, start a static server:
   - `python3 -m http.server 8000`
2. Open:
   - `http://localhost:8000`

## Run with Docker

1. Build the image:
   - `docker build -t arts-demo .`
2. Run it:
   - `docker run --rm -p 8080:80 arts-demo`
3. Open:
   - `http://localhost:8080`

## Current deployed endpoint

- `app_url`: `http://arts-demo-alb-1500855543.us-east-1.elb.amazonaws.com`
- `alb_dns_name`: `arts-demo-alb-1500855543.us-east-1.elb.amazonaws.com`
- `elb_url`: `http://arts-demo-alb-1500855543.us-east-1.elb.amazonaws.com`
- `ecr_repository_url`: `466956024587.dkr.ecr.us-east-1.amazonaws.com/arts-demo-web`