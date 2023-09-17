$env:DOCKER_BUILDKIT=0;
$env:COMPOSE_DOCKER_CLI_BUILD=0;
docker build -t mpazdan/drmdockerfile .
docker run --rm -it `
-p 8000:80 `
-p 44353:443 `
-e ASPNETCORE_ENVIRONMENT=Development `
-e ASPNETCORE_URLS="https://+;http://+" `
-e ASPNETCORE_HTTPS_PORT=44353 `
-e ASPNETCORE_Kestrel__Certificates__Default__Password=marysia `
-e ASPNETCORE_Kestrel__Certificates__Default__Path=/https/aspnetapp.pfx `
-v $env:USERPROFILE\.aspnet\https:/https/ `
--name drmdockercontainer `
mpazdan/drmdockerfile
