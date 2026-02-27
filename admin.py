#!/usr/bin/env python3
"""
Enhanced Face Recognition System Administration Tool

This tool provides comprehensive administrative functions for managing the face recognition system:
- Database management (users, embeddings, metadata)
- Enrollment folder management
- Project data cleaning (face photos, action history, cache files)
- System maintenance (backups, cleanup, optimization)
- Storage analysis and management
- Cascading deletion (user + all associated data)

Usage:
    python admin_merged.py

Features:
    - Interactive menu system with advanced options
    - Safe deletion with confirmation and backups
    - Comprehensive data cleaning capabilities
    - Enrollment folder integration
    - Cascading user deletion (removes all user data)
    - Storage usage analysis
    - Batch operations
    - System health checks
"""

import json
import shutil
import glob
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Import existing modules
try:
    from src.recognize import load_db_npz
except ImportError:
    print("Warning: Could not import face recognition modules")


class ProjectDataCleaner:
    """Handles cleaning of various project data files"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.history_dir = self.project_root / "history"
        self.models_dir = self.project_root / "models"
        self.enroll_dir = self.project_root / "data" / "enroll"
        self.cache_dirs = [
            self.project_root / ".windsurf",
            self.project_root / ".ruff_cache",
            self.project_root / ".pytest_cache",
            self.project_root / "__pycache__",
        ]
        
        print(f"Project Data Cleaner initialized")
        print(f"Project root: {self.project_root}")
    
    def analyze_storage(self) -> Dict[str, Any]:
        """Analyze storage usage across the project"""
        analysis = {
            'total_size': 0,
            'directories': {},
            'large_files': [],
            'old_files': []
        }
        
        # Analyze main directories
        directories_to_check = [
            ("data", self.data_dir),
            ("history", self.history_dir), 
            ("models", self.models_dir),
            ("src", self.project_root / "src"),
            ("enroll", self.enroll_dir),
        ]
        
        for name, path in directories_to_check:
            if path.exists():
                size, file_count = self._get_directory_size(path)
                analysis['directories'][name] = {
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'file_count': file_count,
                    'path': str(path)
                }
                analysis['total_size'] += size
        
        # Find large files (>10MB)
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:
                    analysis['large_files'].append({
                        'path': str(file_path),
                        'size_mb': file_path.stat().st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
        
        # Find old files (>30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    modified_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if modified_date < cutoff_date:
                        analysis['old_files'].append({
                            'path': str(file_path),
                            'size_mb': file_path.stat().st_size / (1024 * 1024),
                            'modified': modified_date,
                            'days_old': (datetime.now() - modified_date).days
                        })
        
        return analysis
    
    def _get_directory_size(self, path: Path) -> Tuple[int, int]:
        """Get total size and file count of directory"""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return total_size, file_count
    
    def delete_user_data_cascade(self, username: str, dry_run: bool = True) -> Dict[str, Any]:
        """Delete all data associated with a user (cascade deletion)"""
        results = {
            'deleted_files': [],
            'deleted_dirs': [],
            'deleted_size': 0,
            'errors': [],
            'found_data': []
        }
        
        print(f"Searching for all data associated with user: {username}")
        
        # 1. Delete enrollment folder
        enroll_user_dir = self.enroll_dir / username
        if enroll_user_dir.exists():
            size, count = self._get_directory_size(enroll_user_dir)
            results['found_data'].append(f"Enrollment folder: {enroll_user_dir} ({size/1024:.1f} KB, {count} files)")
            
            if not dry_run:
                try:
                    shutil.rmtree(enroll_user_dir)
                    results['deleted_dirs'].append(str(enroll_user_dir))
                    results['deleted_size'] += size
                    print(f"✓ Deleted enrollment folder: {enroll_user_dir}")
                except Exception as e:
                    results['errors'].append(f"Error deleting enrollment folder: {e}")
            else:
                results['deleted_dirs'].append(f"[DRY RUN] {enroll_user_dir}")
                results['deleted_size'] += size
        
        # 2. Delete face photos in various locations
        face_photo_patterns = [
            self.data_dir / "faces" / f"{username}*",
            self.data_dir / "enrollments" / f"{username}*",
            self.data_dir / "photos" / f"{username}*",
            self.project_root / "faces" / f"{username}*",
            self.enroll_dir / username / "**" / f"{username}*",
        ]
        
        for pattern in face_photo_patterns:
            for file_path in self.project_root.glob(str(pattern)):
                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        results['found_data'].append(f"Face photo: {file_path}")
                        
                        if not dry_run:
                            file_path.unlink()
                            results['deleted_files'].append(str(file_path))
                            results['deleted_size'] += file_size
                            print(f"✓ Deleted face photo: {file_path}")
                        else:
                            results['deleted_files'].append(f"[DRY RUN] {file_path}")
                            results['deleted_size'] += file_size
                    except Exception as e:
                        results['errors'].append(f"Error deleting {file_path}: {e}")
        
        # 3. Delete action history files for this user
        for history_file in self.history_dir.glob(f"*{username}*.json"):
            try:
                file_size = history_file.stat().st_size
                results['found_data'].append(f"Action history: {history_file}")
                
                if not dry_run:
                    history_file.unlink()
                    results['deleted_files'].append(str(history_file))
                    results['deleted_size'] += file_size
                    print(f"✓ Deleted action history: {history_file}")
                else:
                    results['deleted_files'].append(f"[DRY RUN] {history_file}")
                    results['deleted_size'] += file_size
            except Exception as e:
                results['errors'].append(f"Error deleting {history_file}: {e}")
        
        # 4. Search for any other files with username in name
        common_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.npy', '.npz', '.json', '.txt']
        for ext in common_extensions:
            for file_path in self.project_root.rglob(f"*{username}*{ext}"):
                if file_path.is_file() and file_path.parent != self.backup_dir:
                    try:
                        file_size = file_path.stat().st_size
                        results['found_data'].append(f"Other file: {file_path}")
                        
                        if not dry_run:
                            file_path.unlink()
                            results['deleted_files'].append(str(file_path))
                            results['deleted_size'] += file_size
                            print(f"✓ Deleted other file: {file_path}")
                        else:
                            results['deleted_files'].append(f"[DRY RUN] {file_path}")
                            results['deleted_size'] += file_size
                    except Exception as e:
                        results['errors'].append(f"Error deleting {file_path}: {e}")
        
        return results
    
    def clean_face_photos(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean face enrollment photos"""
        results = {
            'deleted_files': [],
            'deleted_size': 0,
            'errors': []
        }
        
        # Common face photo directories
        face_photo_dirs = [
            self.data_dir / "faces",
            self.data_dir / "enrollments", 
            self.data_dir / "photos",
            self.project_root / "faces",
            self.enroll_dir,
        ]
        
        for photo_dir in face_photo_dirs:
            if not photo_dir.exists():
                continue
            
            print(f"Scanning {photo_dir}...")
            
            for file_path in photo_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        file_size = file_path.stat().st_size
                        
                        # Check if corresponding user exists in database
                        user_name = file_path.stem
                        if self._user_exists_in_db(user_name):
                            continue  # Keep photos for existing users
                        
                        if not dry_run:
                            file_path.unlink()
                            results['deleted_files'].append(str(file_path))
                            results['deleted_size'] += file_size
                        else:
                            results['deleted_files'].append(f"[DRY RUN] {file_path}")
                            results['deleted_size'] += file_size
                            
                    except Exception as e:
                        results['errors'].append(f"Error processing {file_path}: {e}")
        
        return results
    
    def clean_enrollment_folders(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean enrollment folders for users not in database"""
        results = {
            'deleted_dirs': [],
            'deleted_size': 0,
            'errors': []
        }
        
        if not self.enroll_dir.exists():
            return results
        
        print(f"Scanning enrollment folders in {self.enroll_dir}...")
        
        for user_dir in self.enroll_dir.iterdir():
            if user_dir.is_dir():
                try:
                    user_name = user_dir.name
                    
                    # Check if user exists in database
                    if self._user_exists_in_db(user_name):
                        continue  # Keep folder for existing users
                    
                    # Get directory size before deletion
                    size, count = self._get_directory_size(user_dir)
                    
                    if not dry_run:
                        shutil.rmtree(user_dir)
                        results['deleted_dirs'].append(str(user_dir))
                        results['deleted_size'] += size
                        print(f"✓ Deleted enrollment folder: {user_dir} ({size/1024:.1f} KB, {count} files)")
                    else:
                        results['deleted_dirs'].append(f"[DRY RUN] {user_dir}")
                        results['deleted_size'] += size
                        print(f"[DRY RUN] Would delete enrollment folder: {user_dir} ({size/1024:.1f} KB, {count} files)")
                        
                except Exception as e:
                    results['errors'].append(f"Error processing {user_dir}: {e}")
        
        return results
    
    def clean_action_history(self, days_to_keep: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """Clean old action history files"""
        results = {
            'deleted_files': [],
            'deleted_size': 0,
            'errors': []
        }
        
        if not self.history_dir.exists():
            return results
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file_path in self.history_dir.glob("*.json"):
            try:
                modified_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if modified_date < cutoff_date:
                    file_size = file_path.stat().st_size
                    
                    if not dry_run:
                        file_path.unlink()
                        results['deleted_files'].append(str(file_path))
                        results['deleted_size'] += file_size
                    else:
                        results['deleted_files'].append(f"[DRY RUN] {file_path}")
                        results['deleted_size'] += file_size
                        
            except Exception as e:
                results['errors'].append(f"Error processing {file_path}: {e}")
        
        return results
    
    def clean_cache_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean cache and temporary files"""
        results = {
            'deleted_files': [],
            'deleted_size': 0,
            'deleted_dirs': [],
            'errors': []
        }
        
        # Clean cache directories
        for cache_dir in self.cache_dirs:
            if cache_dir.exists():
                try:
                    if not dry_run:
                        shutil.rmtree(cache_dir)
                        results['deleted_dirs'].append(str(cache_dir))
                    else:
                        results['deleted_dirs'].append(f"[DRY RUN] {cache_dir}")
                except Exception as e:
                    results['errors'].append(f"Error deleting {cache_dir}: {e}")
        
        # Clean Python cache files
        for file_path in self.project_root.rglob("*.pyc"):
            try:
                file_size = file_path.stat().st_size
                if not dry_run:
                    file_path.unlink()
                    results['deleted_files'].append(str(file_path))
                    results['deleted_size'] += file_size
                else:
                    results['deleted_files'].append(f"[DRY RUN] {file_path}")
                    results['deleted_size'] += file_size
            except Exception as e:
                results['errors'].append(f"Error deleting {file_path}: {e}")
        
        # Clean __pycache__ directories
        for pycache_dir in self.project_root.rglob("__pycache__"):
            try:
                if pycache_dir.is_dir():
                    if not dry_run:
                        shutil.rmtree(pycache_dir)
                        results['deleted_dirs'].append(str(pycache_dir))
                    else:
                        results['deleted_dirs'].append(f"[DRY RUN] {pycache_dir}")
            except Exception as e:
                results['errors'].append(f"Error deleting {pycache_dir}: {e}")
        
        return results
    
    def clean_log_files(self, days_to_keep: int = 7, dry_run: bool = True) -> Dict[str, Any]:
        """Clean old log files"""
        results = {
            'deleted_files': [],
            'deleted_size': 0,
            'errors': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Common log file patterns
        log_patterns = [
            "*.log",
            "*.log.*", 
            "logs/*.log",
            "*.out",
            "*.err"
        ]
        
        for pattern in log_patterns:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        modified_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if modified_date < cutoff_date:
                            file_size = file_path.stat().st_size
                            
                            if not dry_run:
                                file_path.unlink()
                                results['deleted_files'].append(str(file_path))
                                results['deleted_size'] += file_size
                            else:
                                results['deleted_files'].append(f"[DRY RUN] {file_path}")
                                results['deleted_size'] += file_size
                                
                except Exception as e:
                    results['errors'].append(f"Error processing {file_path}: {e}")
        
        return results
    
    def clean_temp_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean temporary files"""
        results = {
            'deleted_files': [],
            'deleted_size': 0,
            'errors': []
        }
        
        # Common temp file patterns
        temp_patterns = [
            "*.tmp",
            "*.temp", 
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        
                        if not dry_run:
                            file_path.unlink()
                            results['deleted_files'].append(str(file_path))
                            results['deleted_size'] += file_size
                        else:
                            results['deleted_files'].append(f"[DRY RUN] {file_path}")
                            results['deleted_size'] += file_size
                            
                except Exception as e:
                    results['errors'].append(f"Error processing {file_path}: {e}")
        
        return results
    
    def _user_exists_in_db(self, username: str) -> bool:
        """Check if user exists in face database"""
        try:
            db_path = self.data_dir / "db" / "face_db.npz"
            if not db_path.exists():
                return False
            
            db = load_db_npz(db_path)
            return username in db
        except:
            return False
    
    def comprehensive_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """Perform comprehensive cleanup of all data types"""
        results = {
            'face_photos': self.clean_face_photos(dry_run),
            'enrollment_folders': self.clean_enrollment_folders(dry_run),
            'action_history': self.clean_action_history(dry_run=dry_run),
            'cache_files': self.clean_cache_files(dry_run),
            'log_files': self.clean_log_files(dry_run=dry_run),
            'temp_files': self.clean_temp_files(dry_run),
            'total_deleted_size': 0,
            'total_deleted_files': 0
        }
        
        # Calculate totals
        for category in ['face_photos', 'enrollment_folders', 'action_history', 'cache_files', 'log_files', 'temp_files']:
            results['total_deleted_size'] += results[category].get('deleted_size', 0)
            results['total_deleted_files'] += len(results[category].get('deleted_files', []))
            results['total_deleted_files'] += len(results[category].get('deleted_dirs', []))
        
        return results


class EnhancedFaceDBAdmin:
    """Enhanced administrative interface for face recognition database"""
    
    def __init__(self, db_path: str = "data/db/face_db.npz", 
                 json_path: str = "data/db/face_db.json"):
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)
        self.backup_dir = Path("data/db/backups")
        self.cleaner = ProjectDataCleaner()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Enhanced Face DB Admin initialized")
        print(f"Database: {self.db_path}")
        print(f"JSON metadata: {self.json_path}")
        print(f"Enrollment directory: {self.cleaner.enroll_dir}")
    
    def load_database(self) -> Dict[str, Any]:
        """Load the face database"""
        try:
            db = load_db_npz(self.db_path)
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load JSON metadata if available"""
        try:
            if self.json_path.exists():
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    def backup_database(self) -> str:
        """Create backup of current database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"face_db_backup_{timestamp}"
        
        # Backup NPZ file
        if self.db_path.exists():
            backup_npz = self.backup_dir / f"{backup_name}.npz"
            shutil.copy2(self.db_path, backup_npz)
            print(f"Database backed up to: {backup_npz}")
        
        # Backup JSON file if exists
        if self.json_path.exists():
            backup_json = self.backup_dir / f"{backup_name}.json"
            shutil.copy2(self.json_path, backup_json)
            print(f"Metadata backed up to: {backup_json}")
        
        return backup_name
    
    def list_users(self) -> None:
        """List all enrolled users with statistics"""
        db = self.load_database()
        metadata = self.load_metadata()
        
        if not db:
            print("No users found in database.")
            return
        
        print("\n" + "="*60)
        print("ENROLLED USERS")
        print("="*60)
        
        for i, (name, embeddings) in enumerate(db.items(), 1):
            # Get embedding count
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    embed_count = 1
                else:
                    embed_count = embeddings.shape[0]
            else:
                embed_count = len(embeddings) if embeddings else 0
            
            print(f"{i:2d}. {name}")
            print(f"    Embeddings: {embed_count}")
            
            # Check enrollment folder
            enroll_dir = self.cleaner.enroll_dir / name
            if enroll_dir.exists():
                size, count = self.cleaner._get_directory_size(enroll_dir)
                print(f"    Enrollment folder: {count} files ({size/1024:.1f} KB)")
            else:
                print(f"    Enrollment folder: Not found")
            
            # Show metadata if available
            if name in metadata:
                user_meta = metadata[name]
                if 'enrollment_date' in user_meta:
                    print(f"    Enrolled: {user_meta['enrollment_date']}")
                if 'image_count' in user_meta:
                    print(f"    Images: {user_meta['image_count']}")
                if 'notes' in user_meta:
                    print(f"    Notes: {user_meta['notes']}")
            
            print()
        
        print(f"Total users: {len(db)}")
    
    def delete_user_cascade(self, username: str) -> bool:
        """Delete a user and all associated data (cascade deletion)"""
        db = self.load_database()
        
        if username not in db:
            print(f"User '{username}' not found in database.")
            return False
        
        # Show user info before deletion
        print(f"\nUser to delete: {username}")
        embeddings = db[username]
        if isinstance(embeddings, np.ndarray):
            embed_count = embeddings.shape[0] if embeddings.ndim > 1 else 1
        else:
            embed_count = len(embeddings) if embeddings else 0
        print(f"Embeddings: {embed_count}")
        
        # Show associated data
        print("\nAssociated data that will be deleted:")
        cascade_results = self.cleaner.delete_user_data_cascade(username, dry_run=True)
        
        if cascade_results['found_data']:
            for data in cascade_results['found_data']:
                print(f"  - {data}")
        else:
            print("  No additional data found")
        
        # Confirmation
        print(f"\n⚠️  This will delete the user and ALL associated data shown above!")
        confirm = input(f"Are you sure you want to delete '{username}' and all associated data? (type 'DELETE' to confirm): ").strip()
        
        if confirm != 'DELETE':
            print("Deletion cancelled.")
            return False
        
        # Create backup before deletion
        backup_name = self.backup_database()
        
        try:
            # 1. Delete from database
            del db[username]
            np.savez_compressed(self.db_path, **db)
            print(f"✓ Deleted user from database: {username}")
            
            # 2. Update JSON metadata if exists
            metadata = self.load_metadata()
            if username in metadata:
                del metadata[username]
                with open(self.json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✓ Deleted metadata for: {username}")
            
            # 3. Delete all associated data
            cascade_results = self.cleaner.delete_user_data_cascade(username, dry_run=False)
            
            print(f"\n✓ User '{username}' and all associated data deleted successfully.")
            print(f"✓ Backup created: {backup_name}")
            print(f"✓ Deleted {len(cascade_results['deleted_files'])} files")
            print(f"✓ Deleted {len(cascade_results['deleted_dirs'])} directories")
            print(f"✓ Freed {cascade_results['deleted_size'] / (1024*1024):.1f} MB")
            
            if cascade_results['errors']:
                print(f"⚠️  Errors encountered: {len(cascade_results['errors'])}")
                for error in cascade_results['errors']:
                    print(f"    {error}")
            
            return True
            
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete a user from the database (original method)"""
        db = self.load_database()
        
        if username not in db:
            print(f"User '{username}' not found in database.")
            return False
        
        # Show user info before deletion
        print(f"\nUser to delete: {username}")
        embeddings = db[username]
        if isinstance(embeddings, np.ndarray):
            embed_count = embeddings.shape[0] if embeddings.ndim > 1 else 1
        else:
            embed_count = len(embeddings) if embeddings else 0
        print(f"Embeddings: {embed_count}")
        
        # Confirmation
        confirm = input(f"\nAre you sure you want to delete '{username}'? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Deletion cancelled.")
            return False
        
        # Create backup before deletion
        backup_name = self.backup_database()
        
        try:
            # Remove from database
            del db[username]
            
            # Save updated database
            np.savez_compressed(self.db_path, **db)
            
            # Update JSON metadata if exists
            metadata = self.load_metadata()
            if username in metadata:
                del metadata[username]
                with open(self.json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"✓ User '{username}' deleted successfully.")
            print(f"✓ Backup created: {backup_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    def get_database_stats(self) -> None:
        """Show database statistics"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        print("\n" + "="*40)
        print("DATABASE STATISTICS")
        print("="*40)
        
        total_users = len(db)
        total_embeddings = 0
        embedding_sizes = []
        users_with_enroll_folders = 0
        
        for name, embeddings in db.items():
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    count = 1
                    size = embeddings.shape[0]
                else:
                    count = embeddings.shape[0]
                    size = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
            else:
                count = len(embeddings) if embeddings else 0
                size = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            
            total_embeddings += count
            if size > 0:
                embedding_sizes.append(size)
            
            # Check enrollment folder
            enroll_dir = self.cleaner.enroll_dir / name
            if enroll_dir.exists():
                users_with_enroll_folders += 1
        
        print(f"Total users: {total_users}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Average embeddings per user: {total_embeddings/total_users:.1f}")
        print(f"Users with enrollment folders: {users_with_enroll_folders}")
        
        if embedding_sizes:
            print(f"Embedding dimension: {embedding_sizes[0]}")
        
        # File sizes
        if self.db_path.exists():
            db_size = self.db_path.stat().st_size / 1024  # KB
            print(f"Database file size: {db_size:.1f} KB")
        
        if self.json_path.exists():
            json_size = self.json_path.stat().st_size / 1024  # KB
            print(f"Metadata file size: {json_size:.1f} KB")
        
        # Enrollment folder size
        if self.cleaner.enroll_dir.exists():
            enroll_size, enroll_count = self.cleaner._get_directory_size(self.cleaner.enroll_dir)
            print(f"Enrollment folder size: {enroll_size/1024:.1f} KB ({enroll_count} files)")
    
    def search_user(self, query: str) -> None:
        """Search for users by name"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        matches = [name for name in db.keys() if query.lower() in name.lower()]
        
        if not matches:
            print(f"No users found matching '{query}'")
            return
        
        print(f"\nUsers matching '{query}':")
        for name in matches:
            embeddings = db[name]
            embed_count = embeddings.shape[0] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else 1
            
            # Check enrollment folder
            enroll_dir = self.cleaner.enroll_dir / name
            has_folder = "✓" if enroll_dir.exists() else "✗"
            
            print(f"  - {name} ({embed_count} embeddings) {has_folder}")
    
    def cleanup_database(self) -> None:
        """Clean up corrupted or invalid entries"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        print("Checking database integrity...")
        
        corrupted_users = []
        valid_users = {}
        
        for name, embeddings in db.items():
            try:
                # Check if embeddings are valid
                if isinstance(embeddings, np.ndarray):
                    if embeddings.size == 0:
                        corrupted_users.append(name)
                        continue
                    # Try to access the data
                    _ = embeddings.shape
                    valid_users[name] = embeddings
                else:
                    if not embeddings or len(embeddings) == 0:
                        corrupted_users.append(name)
                        continue
                    valid_users[name] = embeddings
                        
            except Exception as e:
                print(f"Corrupted entry found: {name} - {e}")
                corrupted_users.append(name)
        
        if not corrupted_users:
            print("✓ Database is clean - no corrupted entries found.")
            return
        
        print(f"\nFound {len(corrupted_users)} corrupted entries:")
        for name in corrupted_users:
            print(f"  - {name}")
        
        confirm = input(f"\nRemove {len(corrupted_users)} corrupted entries? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
        
        # Create backup
        backup_name = self.backup_database()
        
        try:
            # Save cleaned database
            np.savez_compressed(self.db_path, **valid_users)
            print(f"✓ Removed {len(corrupted_users)} corrupted entries.")
            print(f"✓ Backup created: {backup_name}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def show_storage_analysis(self) -> None:
        """Show comprehensive storage analysis"""
        print("\n" + "="*60)
        print("STORAGE ANALYSIS")
        print("="*60)
        
        analysis = self.cleaner.analyze_storage()
        
        print(f"Total project size: {analysis['total_size'] / (1024*1024):.1f} MB")
        print()
        
        print("Directory breakdown:")
        for name, info in analysis['directories'].items():
            print(f"  {name}/: {info['size_mb']:.1f} MB ({info['file_count']} files)")
        
        if analysis['large_files']:
            print(f"\nLarge files (>10MB):")
            for file_info in analysis['large_files'][:5]:  # Show top 5
                print(f"  {file_info['path']}: {file_info['size_mb']:.1f} MB")
        
        if analysis['old_files']:
            print(f"\nOld files (>30 days): {len(analysis['old_files'])} files")
    
    def interactive_cleanup_menu(self) -> None:
        """Interactive data cleaning menu"""
        while True:
            print("\n" + "="*50)
            print("DATA CLEANING MENU")
            print("="*50)
            print("1. Face photos cleanup")
            print("2. Enrollment folders cleanup")
            print("3. Action history cleanup")
            print("4. Cache files cleanup")
            print("5. Log files cleanup")
            print("6. Temporary files cleanup")
            print("7. Comprehensive cleanup (all)")
            print("8. Storage analysis")
            print("9. Back to main menu")
            print("-" * 50)
            
            try:
                choice = input("Select option (1-9): ").strip()
                
                if choice == '1':
                    self._cleanup_face_photos()
                elif choice == '2':
                    self._cleanup_enrollment_folders()
                elif choice == '3':
                    self._cleanup_action_history()
                elif choice == '4':
                    self._cleanup_cache_files()
                elif choice == '5':
                    self._cleanup_log_files()
                elif choice == '6':
                    self._cleanup_temp_files()
                elif choice == '7':
                    self._comprehensive_cleanup()
                elif choice == '8':
                    self.show_storage_analysis()
                elif choice == '9':
                    break
                else:
                    print("Invalid option. Please select 1-9.")
                    
            except KeyboardInterrupt:
                print("\n\nReturning to main menu...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _cleanup_face_photos(self) -> None:
        """Clean face photos with user interaction"""
        print("\nFace Photos Cleanup")
        print("-" * 30)
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_face_photos(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_files'])} files")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            if results['deleted_files']:
                print("\nFiles to be deleted:")
                for file_path in results['deleted_files'][:5]:  # Show first 5
                    print(f"  {file_path}")
                if len(results['deleted_files']) > 5:
                    print(f"  ... and {len(results['deleted_files']) - 5} more")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_face_photos(dry_run=False)
                print(f"✓ Deleted {len(results['deleted_files'])} files")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_files'])} files")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _cleanup_enrollment_folders(self) -> None:
        """Clean enrollment folders with user interaction"""
        print("\nEnrollment Folders Cleanup")
        print("-" * 30)
        print("This will remove enrollment folders for users not in the database.")
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_enrollment_folders(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_dirs'])} folders")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            if results['deleted_dirs']:
                print("\nFolders to be deleted:")
                for folder_path in results['deleted_dirs']:
                    print(f"  {folder_path}")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_enrollment_folders(dry_run=False)
                print(f"✓ Deleted {len(results['deleted_dirs'])} folders")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_dirs'])} folders")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _cleanup_action_history(self) -> None:
        """Clean action history with user interaction"""
        print("\nAction History Cleanup")
        print("-" * 30)
        
        days = input("Keep history for how many days? (default: 30): ").strip()
        days_to_keep = int(days) if days.isdigit() else 30
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_action_history(days_to_keep=days_to_keep, dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_files'])} files")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_action_history(days_to_keep=days_to_keep, dry_run=False)
                print(f"✓ Deleted {len(results['deleted_files'])} files")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_files'])} files")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _cleanup_cache_files(self) -> None:
        """Clean cache files"""
        print("\nCache Files Cleanup")
        print("-" * 30)
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_cache_files(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_files'])} files")
            print(f"[Dry Run] Would delete {len(results['deleted_dirs'])} directories")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_cache_files(dry_run=False)
                print(f"✓ Deleted {len(results['deleted_files'])} files")
                print(f"✓ Deleted {len(results['deleted_dirs'])} directories")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_files'])} files")
            print(f"✓ Deleted {len(results['deleted_dirs'])} directories")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _cleanup_log_files(self) -> None:
        """Clean log files"""
        print("\nLog Files Cleanup")
        print("-" * 30)
        
        days = input("Keep logs for how many days? (default: 7): ").strip()
        days_to_keep = int(days) if days.isdigit() else 7
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_log_files(days_to_keep=days_to_keep, dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_files'])} files")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_log_files(days_to_keep=days_to_keep, dry_run=False)
                print(f"✓ Deleted {len(results['deleted_files'])} files")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_files'])} files")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _cleanup_temp_files(self) -> None:
        """Clean temporary files"""
        print("\nTemporary Files Cleanup")
        print("-" * 30)
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.clean_temp_files(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {len(results['deleted_files'])} files")
            print(f"[Dry Run] Would free {results['deleted_size'] / (1024*1024):.1f} MB")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.clean_temp_files(dry_run=False)
                print(f"✓ Deleted {len(results['deleted_files'])} files")
                print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {len(results['deleted_files'])} files")
            print(f"✓ Freed {results['deleted_size'] / (1024*1024):.1f} MB")
    
    def _comprehensive_cleanup(self) -> None:
        """Comprehensive cleanup of all data types"""
        print("\nComprehensive Cleanup")
        print("-" * 30)
        print("This will clean ALL data types (photos, enrollment folders, history, cache, logs, temp files)")
        
        dry_run = input("Perform dry run first? (y/n): ").lower().strip() != 'n'
        
        results = self.cleaner.comprehensive_cleanup(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[Dry Run] Would delete {results['total_deleted_files']} files/folders")
            print(f"[Dry Run] Would free {results['total_deleted_size'] / (1024*1024):.1f} MB")
            
            print("\nBreakdown by category:")
            for category, data in results.items():
                if category not in ['total_deleted_files', 'total_deleted_size'] and isinstance(data, dict):
                    file_count = len(data.get('deleted_files', [])) + len(data.get('deleted_dirs', []))
                    print(f"  {category}: {file_count} files/folders")
            
            proceed = input("\nProceed with actual deletion? (y/n): ").lower().strip()
            if proceed == 'y':
                results = self.cleaner.comprehensive_cleanup(dry_run=False)
                print(f"✓ Deleted {results['total_deleted_files']} files/folders")
                print(f"✓ Freed {results['total_deleted_size'] / (1024*1024):.1f} MB")
        else:
            print(f"✓ Deleted {results['total_deleted_files']} files/folders")
            print(f"✓ Freed {results['total_deleted_size'] / (1024*1024):.1f} MB")
    
    def interactive_menu(self) -> None:
        """Interactive menu system"""
        while True:
            print("\n" + "="*60)
            print("ENHANCED FACE RECOGNITION DATABASE ADMIN")
            print("="*60)
            print("Database Management:")
            print("1. List all users")
            print("2. Delete user (database only)")
            print("3. Delete user (CASCADE - all data)")
            print("4. Search users")
            print("5. Database statistics")
            print("6. Cleanup database")
            print("7. Create backup")
            print("\nData Cleaning:")
            print("8. Data cleaning menu")
            print("9. Storage analysis")
            print("\nSystem:")
            print("10. Exit")
            print("-" * 60)
            
            try:
                choice = input("Select option (1-10): ").strip()
                
                if choice == '1':
                    self.list_users()
                
                elif choice == '2':
                    username = input("Enter username to delete (database only): ").strip()
                    if username:
                        self.delete_user(username)
                    else:
                        print("Invalid username.")
                
                elif choice == '3':
                    username = input("Enter username to delete (CASCADE - all data): ").strip()
                    if username:
                        self.delete_user_cascade(username)
                    else:
                        print("Invalid username.")
                
                elif choice == '4':
                    query = input("Enter search query: ").strip()
                    if query:
                        self.search_user(query)
                    else:
                        print("Invalid query.")
                
                elif choice == '5':
                    self.get_database_stats()
                
                elif choice == '6':
                    self.cleanup_database()
                
                elif choice == '7':
                    backup_name = self.backup_database()
                    print(f"Backup created: {backup_name}")
                
                elif choice == '8':
                    self.interactive_cleanup_menu()
                
                elif choice == '9':
                    self.show_storage_analysis()
                
                elif choice == '10':
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid option. Please select 1-10.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main admin interface"""
    print("Enhanced Face Recognition Database Administration Tool")
    print("=" * 70)
    print("Features: Database management + Enrollment folders + Comprehensive data cleaning")
    print("New: Cascade deletion removes ALL user data (database + folders + photos + history)")
    
    try:
        admin = EnhancedFaceDBAdmin()
        admin.interactive_menu()
        
    except Exception as e:
        print(f"Error initializing admin tool: {e}")


if __name__ == "__main__":
    main()
