# Azure provider version
terraform {
  required_version = ">= 1.3.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "3.111.0"
    }
    modtm = {
      source  = "Azure/modtm"
      version = ">= 0.1.8, < 1.0"
    }
  }
}

provider "azurerm" {
  skip_provider_registration = "true"
  features {}

}

provider "modtm" {
  enabled = false
}

resource "random_string" "suffix" {
  length  = 4
  upper   = false
  special = false
}

resource "azurerm_resource_group" "main" {
  location = "eastus2"
  name     = "rg-${random_string.suffix.result}"
}

module "openai" {
  version                       = "0.1.3"
  source                        = "Azure/openai/azurerm"
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  sku_name                      = "S0"
  public_network_access_enabled = true
  deployment = {
    "gpt-4-turbo" = {
      name            = "gpt-4-turbo"
      rai_policy_name = "Microsoft.Default"
      model_format    = "OpenAI"
      model_name      = "gpt-4"
      model_version   = "turbo-2024-04-09"
      scale_type      = "Standard"
      capacity        = 40
    }
  }
  depends_on = [
    azurerm_resource_group.main
  ]
}