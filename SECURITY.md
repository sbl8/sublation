# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Sublation, please report it privately to the maintainers.

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

1. **Email**: Send details to [your-email@domain.com] with "SECURITY" in the subject line
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Regular Updates**: Every week until resolved
- **Public Disclosure**: After fix is available

### Security Considerations for High-Performance Code

Sublation uses unsafe operations and SIMD instructions for performance. When reporting vulnerabilities:

- **Memory Safety**: Buffer overflows, out-of-bounds access
- **Race Conditions**: Concurrent access to shared data
- **Input Validation**: Malformed models or data
- **Resource Exhaustion**: Memory or compute DoS vectors

## Responsible Disclosure

We ask that you:

- Give us reasonable time to fix the issue before public disclosure
- Avoid accessing, modifying, or deleting data that isn't yours
- Don't perform testing that could degrade service quality

## Recognition

Security researchers who responsibly disclose vulnerabilities will be acknowledged in our security advisories (with permission).
