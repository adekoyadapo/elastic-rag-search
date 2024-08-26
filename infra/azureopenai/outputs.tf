output "keys" {
  value     = module.openai.openai_primary_key
  sensitive = true
}

output "domains" {
  value = module.openai.openai_endpoint
}
output "id" {
  value = module.openai.openai_id
}

# https://azure-openai-194691.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2024-02-01

