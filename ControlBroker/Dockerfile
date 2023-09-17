#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443
EXPOSE 8001

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src

RUN echo lets_get_startet

COPY *.sln .
COPY ["Movement/ControlContracts.csproj", "Movement/"]
COPY Movement/. ./Movement.
COPY ["ControlBroker/ControlBroker.csproj", "ControlBroker/"]
COPY ControlBroker/. ./ControlBroker.
COPY . ./
WORKDIR "/src/ControlBroker"
RUN dotnet build "ControlBroker.csproj" -o /app/build


FROM build AS publish
RUN dotnet publish "ControlBroker.csproj" -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ControlBroker.dll"]
# COPY ["ControlBroker/Movement/ControlContracts.csproj", "./ControlBroker/Movement/"]
# COPY ["./ControlBroker/ControlBroker/ControlBroker.csproj", "./ControlBroker/ControlBroker/"]



# COPY ControlBroker/*.sln ControlBroker/
# COPY ["ControlBroker/Movement/ControlContracts.csproj", "ControlBroker/Movement/"]
# COPY ControlBroker/Movement/. ./ControlBroker/Movement.
# COPY ["ControlBroker/ControlBroker/ControlBroker.csproj", "ControlBroker/ControlBroker/"]
# COPY ./ControlBroker/ControlBroker/. ./ControlBroker/ControlBroker.
# COPY ./ControlBroker/ ./ControlBroker/
# WORKDIR "/src/ControlBroker/ControlBroker"
# RUN dotnet build "ControlBroker.csproj" -o /app/build