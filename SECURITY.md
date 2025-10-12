# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do not** create a public issue
2. Email the maintainers directly
3. Include detailed steps to reproduce the vulnerability
4. Allow reasonable time for the issue to be addressed

## Security Considerations

### Data Privacy
- This project processes facial images and biometric data
- Ensure proper consent and privacy compliance when using with personal data
- Follow applicable data protection regulations (GDPR, CCPA, etc.)

### Model Security
- Trained models may be sensitive to adversarial attacks
- Validate input data before processing
- Consider model robustness in production environments

### Dependencies
- Regularly update dependencies to patch known vulnerabilities
- Review third-party licenses and security advisories
- Use virtual environments to isolate dependencies

### Dataset Usage
- Ensure proper licensing for all datasets used
- Respect dataset usage restrictions and attribution requirements
- Do not commit personal or sensitive data to the repository

## Best Practices

1. **Environment Setup**
   - Use virtual environments
   - Keep dependencies updated
   - Use environment variables for sensitive configuration

2. **Data Handling**
   - Anonymize or pseudonymize personal data
   - Implement proper access controls
   - Follow data retention policies

3. **Code Security**
   - Validate all inputs
   - Use secure coding practices
   - Regular security audits

## Compliance Notes

- This project includes components with non-commercial license restrictions
- Ensure compliance with all third-party licenses
- Review export control regulations if applicable