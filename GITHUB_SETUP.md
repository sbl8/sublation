# GitHub Setup Instructions

This document outlines the steps to properly set up the Sublation project on GitHub with auto-documentation.

## Repository Setup

### 1. Create Private Repository

- Create a new private repository on GitHub
- Initialize with the existing code
- Ensure GitHub Pro is enabled for private repo Pages

### 2. Repository Settings

#### Pages Configuration

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` (will be created automatically)
4. Folder: `/` (root)

#### Actions Configuration

1. Go to Settings → Actions → General
2. Actions permissions: Allow all actions and reusable workflows
3. Workflow permissions: Read and write permissions

#### Branch Protection (Optional)

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks (CI tests)
   - Require up-to-date branches

### 3. Secrets and Variables

No secrets required for basic setup, but consider adding:

- `CODECOV_TOKEN` for coverage reporting
- `DISCORD_WEBHOOK` for notifications

## Auto-Documentation Setup

The project includes GitHub Actions workflows that automatically:

1. **Generate API Documentation** - Uses `pkgsite` to create Go package docs
2. **Build Documentation Site** - Creates a static site with navigation
3. **Deploy to Pages** - Publishes to GitHub Pages on every push

### Workflow Files

- `.github/workflows/docs.yml` - Documentation generation and deployment
- `.github/workflows/ci.yml` - Continuous integration testing

### Manual Trigger

To manually trigger documentation rebuild:

```bash
gh workflow run docs.yml
```

## Local Development

### Prerequisites

```bash
# Install required tools
go install golang.org/x/pkgsite/cmd/pkgsite@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

### Development Workflow

```bash
# Install dependencies
make deps

# Run full development pipeline
make dev

# Build and test
make ci

# Generate documentation locally
make docs

# Serve documentation
make docs-serve
```

### Pre-commit Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] `make build` succeeds
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated

## Documentation URLs

Once set up, documentation will be available at:

- **GitHub Pages**: `https://yourusername.github.io/sublation/`
- **API Docs**: `https://yourusername.github.io/sublation/pkg/`
- **Architecture**: `https://yourusername.github.io/sublation/architecture.md`

## Troubleshooting

### Common Issues

#### Pages Not Deploying

- Check Actions tab for failed workflows
- Ensure Pages is enabled in repository settings
- Verify workflow permissions allow writing

#### Import Path Issues

- Update `go.mod` with correct GitHub repository path
- Run `go mod tidy` after changing module path
- Update documentation workflow if needed

#### Linter Errors

- Check `.golangci.yml` configuration
- Run `golangci-lint run` locally to debug
- Some unsafe operations are intentionally allowed

### Performance Monitoring

- Benchmark results are preserved in CI artifacts
- Use `make profile` for local performance analysis
- Monitor documentation build times in Actions

## Next Steps

1. **Push to GitHub**: Upload the codebase to your private repository
2. **Trigger Actions**: First push will automatically trigger CI and docs
3. **Verify Pages**: Check that documentation site deploys correctly
4. **Customize**: Update repository name in workflows and documentation
