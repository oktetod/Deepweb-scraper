"""
Bulk Document Import Tool for Intelligence Agent System
Easily add multiple documents to your knowledge base
"""

import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "https://your-app-url.modal.run"  # UPDATE THIS!
API_ADD_DOC = f"{API_BASE_URL}/api/add_document"

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "title": "LockBit Ransomware Overview",
        "url": "https://example.com/lockbit",
        "text": """LockBit is a ransomware-as-a-service (RaaS) operation that emerged in September 2019. 
        It is known for its automated encryption processes and affiliate model. LockBit 3.0, released in 2022, 
        introduced enhanced capabilities including the world's first ransomware bug bounty program. The group 
        has targeted organizations across various sectors globally, employing double extortion tactics."""
    },
    {
        "title": "APT29 Threat Profile",
        "url": "https://example.com/apt29",
        "text": """APT29, also known as Cozy Bear, is a sophisticated threat actor believed to be associated 
        with Russian intelligence services. The group has been active since at least 2008 and is known for 
        conducting long-term espionage operations targeting government, diplomatic, and think tank entities. 
        APT29 employs advanced techniques including zero-day exploits, supply chain compromises, and custom malware."""
    },
    {
        "title": "Log4Shell Vulnerability Analysis",
        "url": "https://example.com/log4shell",
        "text": """Log4Shell (CVE-2021-44228) is a critical remote code execution vulnerability in Apache Log4j 
        library discovered in December 2021. The vulnerability allows attackers to execute arbitrary code on 
        affected systems through JNDI injection. It affects millions of applications worldwide and has been 
        actively exploited in the wild. Organizations must patch to Log4j 2.17.1 or later."""
    },
    {
        "title": "Phishing Campaigns 2024 Trends",
        "url": "https://example.com/phishing-2024",
        "text": """Phishing attacks in 2024 have evolved to leverage AI-generated content, deepfake technology, 
        and sophisticated social engineering. Business Email Compromise (BEC) remains a top threat, with 
        attackers using compromised legitimate accounts. Mobile phishing (smishing) has increased by 45%, 
        targeting banking and cryptocurrency users. Multi-factor authentication bypass techniques are becoming 
        more prevalent."""
    },
    {
        "title": "Zero Trust Architecture Guide",
        "url": "https://example.com/zero-trust",
        "text": """Zero Trust is a security framework that eliminates implicit trust and continuously validates 
        every stage of digital interaction. Core principles include: verify explicitly, use least privilege 
        access, and assume breach. Implementation requires identity verification, device compliance checking, 
        micro-segmentation, and continuous monitoring. Organizations adopting Zero Trust report 50% reduction 
        in breach impact."""
    }
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def add_single_document(doc: Dict[str, str], timeout: int = 30) -> bool:
    """Add a single document to the database"""
    try:
        response = requests.post(
            API_ADD_DOC,
            json=doc,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success":
            doc_id = result.get("doc_id", "unknown")
            logger.info(f"✅ Added '{doc['title']}' (ID: {doc_id})")
            return True
        else:
            logger.error(f"❌ Failed to add '{doc['title']}': {result.get('detail')}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout adding '{doc['title']}'")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error adding '{doc['title']}': {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error adding '{doc['title']}': {e}")
        return False

def add_documents_from_list(documents: List[Dict[str, str]], delay: float = 1.0) -> Dict[str, int]:
    """Add multiple documents with delay between requests"""
    stats = {"success": 0, "failed": 0, "total": len(documents)}
    
    logger.info(f"📚 Starting bulk import of {stats['total']} documents...")
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"[{i}/{stats['total']}] Processing: {doc['title']}")
        
        if add_single_document(doc):
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        # Delay between requests to avoid rate limiting
        if i < stats['total']:
            time.sleep(delay)
    
    return stats

def add_documents_from_json(filepath: str, delay: float = 1.0) -> Dict[str, int]:
    """Load and add documents from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if not isinstance(documents, list):
            logger.error("❌ JSON file must contain an array of documents")
            return {"success": 0, "failed": 0, "total": 0}
        
        # Validate document structure
        valid_docs = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"⚠️ Skipping invalid document at index {i}")
                continue
            
            if "text" not in doc or not doc["text"]:
                logger.warning(f"⚠️ Skipping document with empty text at index {i}")
                continue
            
            # Set defaults
            doc.setdefault("title", f"Document {i+1}")
            doc.setdefault("url", "")
            
            valid_docs.append(doc)
        
        logger.info(f"📄 Loaded {len(valid_docs)} valid documents from {filepath}")
        return add_documents_from_list(valid_docs, delay)
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filepath}")
        return {"success": 0, "failed": 0, "total": 0}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format: {e}")
        return {"success": 0, "failed": 0, "total": 0}
    except Exception as e:
        logger.error(f"❌ Error reading file: {e}")
        return {"success": 0, "failed": 0, "total": 0}

def add_sample_documents(delay: float = 1.0) -> Dict[str, int]:
    """Add predefined sample documents"""
    logger.info("📚 Adding sample cybersecurity documents...")
    return add_documents_from_list(SAMPLE_DOCUMENTS, delay)

def create_sample_json(output_path: str = "documents.json"):
    """Create a sample JSON file for bulk import"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_DOCUMENTS, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Created sample JSON file: {output_path}")
        logger.info("Edit this file and run: python add_documents.py --json documents.json")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create sample file: {e}")
        return False

def print_summary(stats: Dict[str, int]):
    """Print import summary"""
    print("\n" + "="*60)
    print("📊 IMPORT SUMMARY")
    print("="*60)
    print(f"Total Documents: {stats['total']}")
    print(f"✅ Successfully Added: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"Success Rate: {(stats['success']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%")
    print("="*60 + "\n")

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bulk Document Import Tool for Intelligence Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add sample cybersecurity documents
  python add_documents.py --sample

  # Add documents from JSON file
  python add_documents.py --json documents.json

  # Create sample JSON template
  python add_documents.py --create-sample

  # Add documents with custom delay
  python add_documents.py --json documents.json --delay 2.0

  # Update API URL
  python add_documents.py --api-url https://your-app.modal.run --sample
        """
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Add predefined sample cybersecurity documents'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        metavar='FILE',
        help='Import documents from JSON file'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample JSON file (documents.json)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        metavar='SECONDS',
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        metavar='URL',
        help='Override API base URL'
    )
    
    args = parser.parse_args()
    
    # Update API URL if provided
    if args.api_url:
        global API_BASE_URL, API_ADD_DOC
        API_BASE_URL = args.api_url.rstrip('/')
        API_ADD_DOC = f"{API_BASE_URL}/api/add_document"
        logger.info(f"🌐 Using API: {API_BASE_URL}")
    
    # Check if API URL is configured
    if "your-app-url" in API_BASE_URL and not args.api_url:
        logger.error("❌ Please configure API_BASE_URL in the script or use --api-url")
        logger.info("Example: python add_documents.py --api-url https://your-app.modal.run --sample")
        return
    
    # Execute based on arguments
    if args.create_sample:
        create_sample_json()
        return
    
    if args.sample:
        stats = add_sample_documents(args.delay)
        print_summary(stats)
    elif args.json:
        stats = add_documents_from_json(args.json, args.delay)
        print_summary(stats)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
