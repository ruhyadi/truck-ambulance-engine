"""Main function."""

import rootutils

ROOT = rootutils.autosetup()

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger()

def main_api(cfg: DictConfig) -> None:
    """Run api server."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.truckamb_api import TruckAmbApi
    from src.api.server import UvicornServer

    log.info(f"Starting API server")

    app = FastAPI(
        title="Truck/Ambulance Detection API",
        description="Truck/Ambulance detection REST API",
        version="1.0.0",
        docs_url="/",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.middleware.cors.allow_origins,
        allow_credentials=cfg.api.middleware.cors.allow_credentials,
        allow_methods=cfg.api.middleware.cors.allow_methods,
        allow_headers=cfg.api.middleware.cors.allow_headers,
    )

    truckamb_api = TruckAmbApi(cfg)
    app.include_router(truckamb_api.router)

    server = UvicornServer(
        app,
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
    )
    server.run()


if __name__ == "__main__":

    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main function."""
        main_api(cfg)

    main()